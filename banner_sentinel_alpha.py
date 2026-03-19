import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_theta_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 200 # Higher lock zone for stability

# KINEMATIC LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 10 
MIN_V, MAX_V = 7.0, 15.0
MIN_VALID_DISPLACEMENT = 40.0 
NUCLEATION_ZONE = 250

class SiblingClusterTheta:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.dist_moved = 0.0
        self.v_window = [] # Store recent velocities for terminal lock
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
        # If validated and high, we "Lock" and only allow Micro-Snaps
        is_locked = self.is_validated and (self.tops[0] < LOCK_ZONE_Y)
        
        target = self.tops[0] - self.v
        matches = [t for t in new_tops if abs(t - target) < 15]
        
        if is_locked:
            # GATED MICRO-SNAP: We only snap if the visual is very close to our momentum
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                # Only snap if correction is < 3 pixels
                if abs(best - target) < 3.0:
                    self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
                else:
                    self.tops = [t - self.v for t in self.tops]
            else:
                self.tops = [t - self.v for t in self.tops]
        else:
            # NORMAL TRACKING
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                
                if MIN_V <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    self.dist_moved += actual_v
                    self.v = (self.v * 0.6) + (actual_v * 0.4)
                    
                    # Log to terminal window (last 5 frames)
                    self.v_window.append(actual_v)
                    if len(self.v_window) > 5: self.v_window.pop(0)
                    
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
                # LOCK TERMINAL VELOCITY: Use average of last 5 good frames
                if len(self.v_window) > 0:
                    self.v = sum(self.v_window) / len(self.v_window)
                for item in self.history: item['valid'] = True

        self._record(f)

        if self.age > 50 and self.consistency_score < 2 and not is_locked: 
            self.active = False
        if self.tops[0] < -60: self.active = False
        return self.active

class SentinelTheta:
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
        if birth: self.clusters.append(SiblingClusterTheta(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_theta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    st = SentinelTheta()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = st.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            if c.is_validated:
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"theta_{i:05}.png"), img_bgr)
    
    pd.DataFrame(st.final_manifest).to_csv("sentinel_theta_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_theta()