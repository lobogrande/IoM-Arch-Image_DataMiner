import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_mu_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 480
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 190

# KINEMATIC & CHROMA LAWS
EXPECTED_V = 9.5 # Updated based on Median forensic data
CONSISTENCY_WINDOW = 8 
MIN_V, MAX_V = 6.0, 15.0
NUCLEATION_ZONE_Y = (250, 460) # Widened birth zone

class SiblingClusterMu:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.v_samples = []
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = [] # Temporary buffer for back-filling
        self.start_frame = frame_idx
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
            # PURE BALLISTIC: No visual snapping allowed to prevent HUD drift
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

        # VALIDATION & BACK-FILLING
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            # CALIBRATE: Use the actual median cruising speed
            if self.v_samples: self.v = np.median(self.v_samples)
            
            # RETROSPECTIVE RECOVERY: Correct the early frames using the calibrated velocity
            # We work backwards from the first 'good' frame to the nucleation frame
            anchor_y = self.tops[0]
            anchor_f = f
            for item in self.history:
                frames_back = anchor_f - item['frame']
                item['y_top'] = anchor_y + (frames_back * self.v)
                item['valid'] = True

        self._record(f)
        if (self.tops[0] + BANNER_H) < 20: self.active = False
        if self.age > 40 and not self.is_validated: self.active = False
        return self.active

class SentinelMu:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def check_structure(self, img_bgr, t):
        h, w, _ = img_bgr.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        row_bgr = img_bgr[y_probe, int(w*0.2):int(w*0.8)]
        # Chroma Filter: Banners are not pure Red
        r_avg, g_avg = np.mean(row_bgr[:, 2]), np.mean(row_bgr[:, 1])
        if r_avg > (g_avg + 40): return False 
        # Texture check
        row_gray = cv2.cvtColor(row_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY)
        return np.var(row_gray) > 10.0

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        center_ints = np.mean(img_gray[:, int(img_gray.shape[1]*0.4):int(img_gray.shape[1]*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_structure(img_bgr, t)]
        
        for c in self.clusters:
            if not c.update(valid_tops, f_idx):
                self.master_history.extend(c.history)
                
        # NUCLEATION: Look for new arrivals in the birth zone
        birth_tops = [t for t in valid_tops if NUCLEATION_ZONE_Y[0] <= t <= NUCLEATION_ZONE_Y[1]]
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 60 for c in self.clusters for t in c.tops)]
        if birth_tops:
            self.clusters.append(SiblingClusterMu(birth_tops, f_idx))
        self.clusters = [c for c in self.clusters if c.active]

    def finalize(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty: return df
        return df[df['valid']].sort_values(['frame', 'id', 'sibling_idx']).drop_duplicates(['frame', 'id', 'sibling_idx'])

def run_sentinel_mu():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    mu = SentinelMu()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: mu.process_frame(img_bgr, i)
    
    manifest = mu.finalize()
    manifest.to_csv("sentinel_mu_manifest.csv", index=False)
    
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
        cv2.imwrite(os.path.join(OUT_DIR, f"mu_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_mu()