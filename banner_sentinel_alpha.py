import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_beta_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
HUD_DANGER_ZONE = 145

# KINEMATIC LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 15 # Stricter vetting to kill ghosts
MIN_V, MAX_V = 7.5, 14.0
NUCLEATION_ZONE = 250

class SiblingClusterBeta:
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
        
        # GUIDED MOMENTUM: Look for a match but reject it if it implies a wild jump
        matches = [t for t in new_tops if abs(t - target) < 25]
        
        best_match = None
        if matches:
            temp_best = min(matches, key=lambda t: abs(t - target))
            actual_v = self.tops[0] - temp_best
            # If the match implies we are stopping or jumping, ignore it (likely the HUD)
            if abs(actual_v - self.v) < 2.5 and MIN_V <= actual_v <= MAX_V:
                best_match = temp_best

        if best_match is not None:
            actual_v = self.tops[0] - best_match
            self.consistency_score += 1
            self.v = (self.v * 0.8) + (actual_v * 0.2)
            self.tops = [best_match] + [best_match + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            # COASTING: Physics-only update
            self.tops = [t - self.v for t in self.tops]
            # Only reduce consistency if we are in the open grid (not the HUD zone)
            if self.tops[0] > HUD_DANGER_ZONE:
                self.consistency_score = max(0, self.consistency_score - 1)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # KILL if it stalls or leaves the screen
        if self.age > 40 and self.consistency_score < 5: self.active = False
        if self.tops[0] < -50: self.active = False
        return self.active

class SentinelBeta:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_sandwich(self, img_gray, t):
        """Verifies the Top-Edge / Text-Core / Black-Bottom structure."""
        h, w = img_gray.shape
        c1, c2 = int(w * 0.3), int(w * 0.7)
        if t + 45 >= h: return False
        
        # 1. CORE VARIANCE (Must have text in the middle)
        core_slice = img_gray[int(t + 22), c1:c2]
        if np.var(core_slice) < 15.0: return False # Grid rows are too uniform
        
        # 2. BOTTOM PADDING (Must be solid dark at the base)
        bottom_slice = img_gray[int(t + 40), c1:c2]
        if np.mean(bottom_slice) > 65 or np.var(bottom_slice) > 30: return False
        
        return True

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_sandwich(img_gray, t)]
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Birth
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 50 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterBeta(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_beta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sb = SentinelBeta()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = sb.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        # --- UI CLEANUP: ONLY DRAW VALIDATED RED BARS ---
        valid_count = 0
        for c in clusters:
            if c.is_validated:
                valid_count += 1
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"BETA F:{i} | VALID BANNERS: {valid_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"beta_{i:05}.png"), img_bgr)
    
    pd.DataFrame(sb.final_manifest).to_csv("sentinel_beta_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_beta()