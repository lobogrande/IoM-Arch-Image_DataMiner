import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = cfg.get_buffer_path(0)
OUT_DIR = "sentinel_pi_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 480
BANNER_H = 45
DRAW_OFFSET = -12 
HUD_BLIND_ZONE = (145, 185) # IGNORE HUD LINES

# KINEMATIC LAWS
UI_SCROLL_V = 10.2 # THE GOLDEN CONSTANT
CONSISTENCY_WINDOW = 10 
MIN_V, MAX_V = 8.0, 12.0 # Tight window around the constant

class SiblingClusterPi:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = UI_SCROLL_V
        self.age = 0
        self.consistency_score = 0
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self._record(frame_idx)

    def _record(self, f):
        for idx, t in enumerate(self.tops):
            self.history.append({
                "frame": f, "id": self.id, "sibling_idx": idx,
                "y_top": float(t), "v": self.v, "valid": self.is_validated
            })

    def update(self, new_tops, f):
        self.age += 1
        # Once validated, we use the UI_SCROLL_V constant
        target_v = UI_SCROLL_V if self.is_validated else self.v
        target_y = self.tops[0] - target_v
        
        # SEARCH GATING: Ignore HUD interference
        matches = [t for t in new_tops if abs(t - target_y) < 15]
        # Filter matches that fall inside the HUD Blind Zone
        matches = [t for t in matches if not (HUD_BLIND_ZONE[0] < t < HUD_BLIND_ZONE[1])]
        
        visual_match = False
        if matches:
            best = min(matches, key=lambda t: abs(t - target_y))
            actual_v = self.tops[0] - best
            if MIN_V <= actual_v <= MAX_V:
                self.consistency_score += 1
                if not self.is_validated: self.v = (self.v * 0.7) + (actual_v * 0.3)
                self.tops = [best] + [best + (self.tops[idx]-self.tops[0]) for idx in range(1, len(self.tops))]
                visual_match = True

        if not visual_match:
            self.tops = [t - target_v for t in self.tops]
            if not self.is_validated: self.consistency_score = max(0, self.consistency_score - 1)

        # VALIDATION & BACKFILL
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            self.v = UI_SCROLL_V # Lock to constant
            anchor_y, anchor_f = self.tops[0], f
            for item in self.history:
                item['y_top'] = anchor_y + ((anchor_f - item['frame']) * UI_SCROLL_V)
                item['valid'] = True

        self._record(f)
        # TERMINATION: Top edge must be off-screen
        if (self.tops[0] + BANNER_H) < 0: self.active = False
        return self.active

class SentinelPi:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def check_structure(self, img_bgr, t):
        h, w, _ = img_bgr.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        row_bgr = img_bgr[y_probe, int(w*0.2):int(w*0.8)]
        r_avg, g_avg = np.mean(row_bgr[:, 2]), np.mean(row_bgr[:, 1])
        if r_avg > (g_avg + 35): return False 
        return True

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        center_ints = np.mean(img_gray[:, int(img_gray.shape[1]*0.4):int(img_gray.shape[1]*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_structure(img_bgr, t)]
        
        for c in self.clusters:
            if not c.update(valid_tops, f_idx):
                self.master_history.extend(c.history)
                
        # NUCLEATION: Tighter spatial filter (35px) for double banners
        birth = [t for t in valid_tops if t > 300]
        birth = [bt for bt in birth if not any(abs(bt - t) < 35 for c in self.clusters for t in c.tops)]
        if birth:
            self.clusters.append(SiblingClusterPi(birth, f_idx))
        self.clusters = [c for c in self.clusters if c.active]

    def finalize(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty: return df
        return df[df['valid']].sort_values(['frame', 'id', 'sibling_idx']).drop_duplicates(['frame', 'id', 'sibling_idx'])

def run_sentinel_pi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    pi = SentinelPi()
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: pi.process_frame(img_bgr, i)
    
    manifest = pi.finalize()
    manifest.to_csv("sentinel_pi_manifest.csv", index=False)
    for i in range(START_F, min(END_F, len(all_files))):
        frame_data = manifest[manifest['frame'] == i]
        if frame_data.empty: continue
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        overlay = img_bgr.copy()
        for _, row in frame_data.iterrows():
            y = int(row['y_top'] + DRAW_OFFSET)
            cv2.rectangle(overlay, (40, y), (1240, y + BANNER_H), (0, 0, 255), -1)
            cv2.putText(img_bgr, f"ID:{int(row['id'])}", (50, y-5), 1, 0.8, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"pi_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_pi()