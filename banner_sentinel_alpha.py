import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_epsilon_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 

# KINEMATIC & SIGNAL LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 15 # Strict probation
MIN_V, MAX_V = 6.0, 16.0
MIN_FILL_RATE = 0.85 # Row must be 85% dark across the screen
NUCLEATION_ZONE = 250

class SiblingClusterEpsilon:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
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
        target = self.tops[0] - self.v
        
        matches = [t for t in new_tops if abs(t - target) < 25]
        best_match = None
        if matches:
            temp_best = min(matches, key=lambda t: abs(t - target))
            actual_v = self.tops[0] - temp_best
            
            # THE SECOND LAW: MUST BE MOVING UP
            if MIN_V <= actual_v <= MAX_V:
                best_match = temp_best
                self.v = (self.v * 0.7) + (actual_v * 0.3)
                self.consistency_score += 1
            else:
                self.consistency_score = max(0, self.consistency_score - 1)

        if best_match is not None:
            self.tops = [best_match] + [best_match + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            # COASTING
            self.tops = [t - self.v for t in self.tops]
            self.consistency_score = max(0, self.consistency_score - 2)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # THE THIRD LAW: STILLNESS KILL
        # If it stops moving or loses consistency, it's noise.
        if self.consistency_score < 1: self.active = False
        if self.tops[0] < -50: self.active = False
        return self.active

class SentinelEpsilon:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_horizontal_fill(self, img_gray, t):
        """THE FIRST LAW: A banner is a full-width entity."""
        h, w = img_gray.shape
        y_probe = int(t + 10)
        if y_probe >= h: return False
        
        # Scan from 10% to 90% of screen width
        x_start, x_end = int(w * 0.1), int(w * 0.9)
        row_segment = img_gray[y_probe, x_start:x_end]
        
        fill_rate = np.mean(row_segment < 60)
        return fill_rate > MIN_FILL_RATE

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_horizontal_fill(img_gray, t)]
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Birth only in nucleation zone
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 45 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterEpsilon(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_epsilon():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    se = SentinelEpsilon()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = se.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        # --- UI CLEANUP: ONLY DRAW VALIDATED RED BOXES ---
        v_count = 0
        for c in clusters:
            if c.is_validated:
                v_count += 1
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"EPSILON F:{i} | VALIDATED: {v_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"eps_{i:05}.png"), img_bgr)
    
    pd.DataFrame(se.final_manifest).to_csv("sentinel_epsilon_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_epsilon()