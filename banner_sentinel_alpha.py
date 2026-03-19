import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_upsilon_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 

# KINEMATIC LAWS
EXPECTED_V = 10.0
MIN_V_ALLOWED = 2.0  # KILL if slower than this
MAX_V_ALLOWED = 18.0 # KILL if faster than this
V_DAMPING = 0.85
STALL_KILL_LIMIT = 4
NUCLEATION_ZONE = 300
TOTAL_DISPLACEMENT_REQUIRED = 150.0 # Banners must travel a long way

class SiblingClusterUpsilon:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.dist_moved = 0.0
        self.stall_count = 0
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
        
        matches = [t for t in new_tops if abs(t - target) < 30]
        if matches:
            best = min(matches, key=lambda t: abs(t - target))
            actual_v = self.tops[0] - best
            
            # --- THE FIRST LAW: UPWARD ONLY ---
            if actual_v < MIN_V_ALLOWED or actual_v > MAX_V_ALLOWED:
                self.stall_count += 1
                # Still move the box predictively but don't 'learn' this bad velocity
                self.tops = [t - EXPECTED_V for t in self.tops]
            else:
                self.stall_count = 0
                self.v = (self.v * V_DAMPING) + (actual_v * (1.0 - V_DAMPING))
                self.dist_moved += actual_v
                self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            # COAST
            self.tops = [t - self.v for t in self.tops]
            self.dist_moved += self.v
            self.stall_count += 0 # Coasting doesn't count as stalling

        # Validation logic
        if not self.is_validated and self.dist_moved >= TOTAL_DISPLACEMENT_REQUIRED:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # Execution Logic
        if self.stall_count > STALL_KILL_LIMIT: self.active = False
        if self.tops[0] < SCAN_Y_START - 50: self.active = False # Left screen
        return self.active

class SentinelUpsilon:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_sandwich(self, img_gray, t):
        """Verifies the Dark-Text-Dark signature of a real banner."""
        h, w = img_gray.shape
        # Sample center 40%
        c1, c2 = int(w*0.3), int(w*0.7)
        if t + 40 >= h: return False
        
        # Layer 1: Top Padding (Should be dark and uniform)
        top_pad = img_gray[t:t+5, c1:c2]
        # Layer 2: Text Core (Should have high horizontal variance)
        core = img_gray[t+10:t+30, c1:c2]
        
        if np.mean(top_pad) < 45 and np.var(core) > 10.0:
            return True
        return False

    def process_frame(self, img_gray, f_idx):
        # 5-Point Horizontal Edge Search
        w = img_gray.shape[1]
        cols = [int(w*0.1), int(w*0.3), int(w*0.5), int(w*0.7), int(w*0.9)]
        ints = np.mean(img_gray[:, cols], axis=1)
        grad = np.diff(ints.astype(float))
        tops = np.where(grad < -7.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_sandwich(img_gray, t)]
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Birth logic (Only in nucleation zone)
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 40 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterUpsilon(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_upsilon():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    su = SentinelUpsilon()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = su.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            if c.is_validated:
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, y_draw), (img_bgr.shape[1], y_draw + BANNER_H), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"up_{i:05}.png"), img_bgr)
    
    pd.DataFrame(su.final_manifest).to_csv("sentinel_upsilon_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_upsilon()