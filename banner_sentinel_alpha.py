import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_xi_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 480
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 195
NUCLEATION_Y = (345, 415)

# KINEMATIC & STRUCTURAL LAWS
EXPECTED_V = 9.8 
CONSISTENCY_WINDOW = 12 
MIN_V, MAX_V = 7.5, 12.5 # Tightened to exclude ghosts
MIN_DENSITY = 0.70

class SiblingClusterXi:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.v_samples = []
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self.start_pos = self.tops[0]
        self._record(frame_idx)

    def _record(self, f):
        for idx, t in enumerate(self.tops):
            self.history.append({
                "frame": f, "id": self.id, "sibling_idx": idx,
                "y_top": float(t), "v": self.v, "valid": self.is_validated
            })

    def update(self, new_tops, f):
        self.age += 1
        is_ballistic = self.is_validated and (self.tops[0] < LOCK_ZONE_Y)
        target = self.tops[0] - self.v
        matches = [t for t in new_tops if abs(t - target) < 15]
        
        visual_match = False
        if is_ballistic:
            self.tops = [t - self.v for t in self.tops]
        else:
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                if MIN_V <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    self.v = (self.v * 0.7) + (actual_v * 0.3)
                    if self.tops[0] > LOCK_ZONE_Y: self.v_samples.append(actual_v)
                    self.tops = [best] + [best + (self.tops[idx]-self.tops[0]) for idx in range(1, len(self.tops))]
                    visual_match = True

        if not visual_match and not is_ballistic:
            self.tops = [t - self.v for t in self.tops]
            self.consistency_score = max(0, self.consistency_score - 1)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            if (self.start_pos - self.tops[0]) > 60: # Banner must move significantly
                self.is_validated = True
                if self.v_samples: self.v = np.median(self.v_samples)
                # Retrospective backfill
                anchor_y, anchor_f = self.tops[0], f
                for item in self.history:
                    item['y_top'] = anchor_y + ((anchor_f - item['frame']) * self.v)
                    item['valid'] = True
            else:
                self.active = False # Kill slow ghosts

        self._record(f)
        if (self.tops[0] + BANNER_H) < 0: self.active = False
        if self.age > 40 and not self.is_validated: self.active = False
        return self.active

class SentinelXi:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def check_sandwich(self, img_bgr, t):
        """Verifies Edge -> Text Core -> Solid Base structure."""
        h, w, _ = img_bgr.shape
        y_core = int(t + 22)
        y_base = int(t + 40)
        if y_base >= h: return False
        
        # 1. Base Layer Check (Solid Black)
        base_row = cv2.cvtColor(img_bgr[y_base, int(w*0.1):int(w*0.9)].reshape(1,-1,3), cv2.COLOR_BGR2GRAY)
        if np.mean(base_row) > 60 or np.var(base_row) > 25: return False
        
        # 2. Text Core Check (High Variance)
        core_row = cv2.cvtColor(img_bgr[y_core, int(w*0.1):int(w*0.9)].reshape(1,-1,3), cv2.COLOR_BGR2GRAY)
        if np.var(core_row) < 15.0: return False # Grid rows are too uniform
        
        # 3. Density Check
        mask = (img_bgr[y_core, :, 0] < 75) & (img_bgr[y_core, :, 1] < 75) & (img_bgr[y_core, :, 2] < 75)
        if np.mean(mask) < MIN_DENSITY: return False
        
        return True

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        center_ints = np.mean(img_gray[:, int(img_gray.shape[1]*0.4):int(img_gray.shape[1]*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        # Use sandwich test for validation
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_sandwich(img_bgr, t)]
        
        for c in self.clusters:
            if not c.update(valid_tops, f_idx):
                self.master_history.extend(c.history)
                
        birth = [t for t in valid_tops if NUCLEATION_Y[0] <= t <= NUCLEATION_Y[1]]
        birth = [bt for bt in birth if not any(abs(bt - t) < 60 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterXi(birth, f_idx))
        self.clusters = [c for c in self.clusters if c.active]

    def finalize(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty or 'valid' not in df.columns: return pd.DataFrame()
        df = df[df['valid']].sort_values(['frame', 'id', 'sibling_idx'])
        return df.drop_duplicates(['frame', 'id', 'sibling_idx'])

def run_sentinel_xi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    xi = SentinelXi()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: xi.process_frame(img_bgr, i)
    
    manifest = xi.finalize()
    if manifest.empty:
        print("AUDIT COMPLETE: No valid banners detected with current structural constraints.")
        return
        
    manifest.to_csv("sentinel_xi_manifest.csv", index=False)
    
    for i in range(START_F, min(END_F, len(all_files))):
        frame_data = manifest[manifest['frame'] == i]
        if frame_data.empty: continue
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        overlay = img_bgr.copy()
        for _, row in frame_data.iterrows():
            y = int(row['y_top'] + DRAW_OFFSET)
            cv2.rectangle(overlay, (40, y), (1240, y + BANNER_H), (0, 0, 255), -1)
            cv2.putText(img_bgr, f"ID:{int(row['id'])} V:{row['v']:.2f}", (50, y-5), 1, 0.8, (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"xi_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_xi()