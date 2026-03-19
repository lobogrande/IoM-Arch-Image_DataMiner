import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_phi_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 

# KINEMATIC LAWS
EXPECTED_V = 10.0
CONSISTENCY_WINDOW = 10 # Frames of consistent motion required
MIN_V, MAX_V = 8.0, 13.0 # Strict "Banner Speed" window
NUCLEATION_ZONE = 250

class SiblingClusterPhi:
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
        if matches:
            best = min(matches, key=lambda t: abs(t - target))
            actual_v = self.tops[0] - best
            
            # Check for "Banner-like" velocity
            if MIN_V <= actual_v <= MAX_V:
                self.consistency_score += 1
                self.v = (self.v * 0.8) + (actual_v * 0.2)
            else:
                self.consistency_score = max(0, self.consistency_score - 2)
            
            self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            # Coasting reduces consistency
            self.tops = [t - self.v for t in self.tops]
            self.consistency_score = max(0, self.consistency_score - 1)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # KILL if it stalls, moves wrong, or leaves screen
        if self.age > 20 and self.consistency_score < 3: self.active = False
        if self.tops[0] < SCAN_Y_START - 50: self.active = False
        return self.active

class SentinelPhi:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_structural_integrity(self, img_gray, t):
        """Verifies horizontal continuity and text contrast."""
        h, w = img_gray.shape
        # 10-Point Horizontal Scan
        cols = np.linspace(w*0.1, w*0.9, 10).astype(int)
        
        # 1. Horizontal Integrity (Must be dark across the bar)
        row_samples = img_gray[t + 5, cols]
        if np.mean(row_samples) > 60: return False
        
        # 2. Contrast Signature (Text exists)
        # Check the vertical slice for high-contrast 'Text' peaks
        core_slice = img_gray[t:t+BANNER_H, int(w*0.5)]
        contrast = np.max(core_slice) - np.min(core_slice)
        if contrast < 100: return False # Text must be bright white vs dark black
        
        return True

    def process_frame(self, img_gray, f_idx):
        # Scan for Top Edges
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_structural_integrity(img_gray, t)]
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Birth logic
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 40 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterPhi(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_phi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sp = SentinelPhi()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = sp.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            # Visual Debug: YELLOW for unvalidated, RED for validated
            color = (0, 0, 255) if c.is_validated else (0, 255, 255)
            for t in c.tops:
                y_draw = int(t + DRAW_OFFSET)
                cv2.rectangle(overlay, (0, y_draw), (img_bgr.shape[1], y_draw + BANNER_H), color, -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"phi_{i:05}.png"), img_bgr)
    
    pd.DataFrame(sp.final_manifest).to_csv("sentinel_phi_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_phi()