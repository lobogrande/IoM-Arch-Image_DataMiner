import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_eta_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 180 # Freeze velocity below this height

# KINEMATIC LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 10 
MIN_V, MAX_V = 7.0, 15.0
MIN_VALID_DISPLACEMENT = 40.0 # Must move this much UP to be valid
NUCLEATION_ZONE = 250

class SiblingClusterEta:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.dist_moved = 0.0
        self.v_sum = 0.0
        self.v_count = 0
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self._record(frame_idx)

    def _record(self, f):
        for t in self.tops:
            self.history.append({"frame": f, "id": self.id, "y": t, "v": self.v, "valid": self.is_validated})

    def update(self, new_tops, f):
        self.age += 1
        
        # 1. INERTIAL LOCK LOGIC
        # If we are validated and high enough, we "Freeze" velocity and ignore input
        is_locked = self.is_validated and (self.tops[0] < LOCK_ZONE_Y)
        
        if is_locked:
            # Use the historical average velocity to sail past the HUD
            self.tops = [t - self.v for t in self.tops]
        else:
            target = self.tops[0] - self.v
            matches = [t for t in new_tops if abs(t - target) < 30]
            
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                
                if MIN_V <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    self.dist_moved += actual_v
                    # Velocity Smoothing
                    self.v = (self.v * 0.7) + (actual_v * 0.3)
                    # For locking, keep track of average velocity
                    self.v_sum += actual_v
                    self.v_count += 1
                    
                    self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
                else:
                    self.consistency_score = max(0, self.consistency_score - 1)
            else:
                self.tops = [t - self.v for t in self.tops]
                self.dist_moved += self.v
                self.consistency_score = max(0, self.consistency_score - 2)

        # TRIGGER VALIDATION: Consistency AND Displacement
        if not self.is_validated:
            if self.consistency_score >= CONSISTENCY_WINDOW and self.dist_moved >= MIN_VALID_DISPLACEMENT:
                self.is_validated = True
                # Set locked velocity to the average seen so far
                if self.v_count > 0: self.v = self.v_sum / self.v_count
                for item in self.history: item['valid'] = True

        self._record(f)

        if self.age > 40 and self.consistency_score < 2 and not is_locked: 
            self.active = False
        if self.tops[0] < -50: self.active = False
        return self.active

class SentinelEta:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_ensemble(self, img_gray, t):
        h, w = img_gray.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        x_start, x_end = int(w * 0.1), int(w * 0.9)
        row_strip = img_gray[y_probe, x_start:x_end]
        fill_rate = np.mean(row_strip < 75)
        variance = np.var(row_strip.astype(float))
        return (fill_rate > 0.55) and (variance > 12.0)

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_ensemble(img_gray, t)]
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 50 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterEta(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_eta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    se = SentinelEta()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = se.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            if c.is_validated:
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"ETA F:{i} | VALIDATED: {len([c for c in clusters if c.is_validated])}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"eta_{i:05}.png"), img_bgr)
    
    pd.DataFrame(se.final_manifest).to_csv("sentinel_eta_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_eta()