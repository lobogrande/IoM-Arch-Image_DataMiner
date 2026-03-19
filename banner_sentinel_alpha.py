import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_nu_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 480
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 195
NUCLEATION_Y = (345, 415) # RESTORED: Rigid birth zone

# KINEMATIC & SIGNAL LAWS
EXPECTED_V = 9.8 
CONSISTENCY_WINDOW = 12 
MIN_V, MAX_V = 5.0, 16.0
MIN_CONTIGUITY_PX = 1000 # Law of the Banner: Must be a full-width bar

class SiblingClusterNu:
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
        matches = [t for t in new_tops if abs(t - target) < 18]
        
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

        # VALIDATION with Displacement Check
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            if (self.start_pos - self.tops[0]) > 50: # The banner MUST move
                self.is_validated = True
                if self.v_samples: self.v = np.median(self.v_samples)
                # Correct the early frames using calibrated velocity
                anchor_y, anchor_f = self.tops[0], f
                for item in self.history:
                    item['y_top'] = anchor_y + ((anchor_f - item['frame']) * self.v)
                    item['valid'] = True
            else:
                self.active = False # Stationary ghost detected

        self._record(f)
        if (self.tops[0] + BANNER_H) < 0: self.active = False
        if self.age > 30 and not self.is_validated and self.consistency_score < 2: self.active = False
        return self.active

class SentinelNu:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def check_contiguity(self, img_bgr, t):
        """Filters out ore rows by requiring a continuous, screen-width black segment."""
        h, w, _ = img_bgr.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        row_bgr = img_bgr[y_probe, :]
        mask = (row_bgr[:, 0] < 65) & (row_bgr[:, 1] < 65) & (row_bgr[:, 2] < 65)
        mask = mask.astype(np.uint8) * 255
        
        # 1D Connected Components to find widest dark block
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.reshape(1, -1), connectivity=8)
        if num_labels <= 1: return False
        return np.max(stats[1:, cv2.CC_STAT_WIDTH]) > MIN_CONTIGUITY_PX

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        center_ints = np.mean(img_gray[:, int(img_gray.shape[1]*0.4):int(img_gray.shape[1]*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_contiguity(img_bgr, t)]
        
        for c in self.clusters:
            if not c.update(valid_tops, f_idx):
                self.master_history.extend(c.history)
                
        birth = [t for t in valid_tops if NUCLEATION_Y[0] <= t <= NUCLEATION_Y[1]]
        birth = [bt for bt in birth if not any(abs(bt - t) < 60 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterNu(birth, f_idx))
        self.clusters = [c for c in self.clusters if c.active]

    def finalize(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty: return df
        df = df[df['valid']].sort_values(['frame', 'id', 'sibling_idx'])
        return df.drop_duplicates(['frame', 'id', 'sibling_idx'])

def run_sentinel_nu():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    nu = SentinelNu()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: nu.process_frame(img_bgr, i)
    
    manifest = nu.finalize()
    manifest.to_csv("sentinel_nu_manifest.csv", index=False)
    
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
        cv2.imwrite(os.path.join(OUT_DIR, f"nu_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_nu()