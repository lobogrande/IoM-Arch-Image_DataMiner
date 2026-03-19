import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_omicron_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 480
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 190
NUCLEATION_ZONE = (340, 420)

# KINEMATIC LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 10 
MIN_V, MAX_V = 6.0, 15.0

class SiblingClusterOmicron:
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

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            if self.v_samples: self.v = np.median(self.v_samples)
            # Correct the early frames using calibrated velocity
            anchor_y, anchor_f = self.tops[0], f
            for item in self.history:
                item['y_top'] = anchor_y + ((anchor_f - item['frame']) * self.v)
                item['valid'] = True

        self._record(f)
        if (self.tops[0] + BANNER_H) < 0: self.active = False
        if self.age > 40 and not self.is_validated: self.active = False
        return self.active

class SentinelOmicron:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def get_structural_score(self, img_bgr, t):
        """Calculates a weighted score for Banner-ness."""
        h, w, _ = img_bgr.shape
        y_core = int(t + 22)
        y_base = int(t + 40)
        if y_base >= h: return 0.0
        
        # 1. Detect Horizontal Bounds (Find the dark segment)
        row_gray = cv2.cvtColor(img_bgr[y_base, :].reshape(1,-1,3), cv2.COLOR_BGR2GRAY).flatten()
        dark_mask = row_gray < 70
        if np.sum(dark_mask) < 300: return 0.0 # Too narrow to be a banner
        
        # 2. Structural Scoring
        core_gray = cv2.cvtColor(img_bgr[y_core, dark_mask].reshape(1,-1,3), cv2.COLOR_BGR2GRAY)
        base_gray = cv2.cvtColor(img_bgr[y_base, dark_mask].reshape(1,-1,3), cv2.COLOR_BGR2GRAY)
        
        # We want High Core Var (Text) and Low Base Var (Uniform Bar)
        text_score = np.var(core_gray)
        base_uniformity = 1.0 / (np.var(base_gray) + 1.0)
        
        # Chroma Penalty: Banners are not Red
        row_bgr = img_bgr[y_core, dark_mask]
        r_bias = np.mean(row_bgr[:, 2]) - np.mean(row_bgr[:, 1])
        chroma_penalty = 1.0 if r_bias < 35 else 0.0
        
        return (text_score * base_uniformity) * chroma_penalty

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        center_ints = np.mean(img_gray[:, int(img_gray.shape[1]*0.4):int(img_gray.shape[1]*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        # Filter candidates by the fuzzy structural score
        valid_candidates = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            score = self.get_structural_score(img_bgr, t)
            if score > 0.8: # Threshold for 'Banner-ness'
                valid_candidates.append(t)
        
        for c in self.clusters:
            if not c.update(valid_candidates, f_idx):
                self.master_history.extend(c.history)
                
        birth = [t for t in valid_candidates if NUCLEATION_ZONE[0] <= t <= NUCLEATION_ZONE[1]]
        birth = [bt for bt in birth if not any(abs(bt - t) < 60 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterOmicron(birth, f_idx))
        self.clusters = [c for c in self.clusters if c.active]

    def finalize(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty or 'valid' not in df.columns: return pd.DataFrame()
        df = df[df['valid']].sort_values(['frame', 'id', 'sibling_idx'])
        return df.drop_duplicates(['frame', 'id', 'sibling_idx'])

def run_sentinel_omicron():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    om = SentinelOmicron()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: om.process_frame(img_bgr, i)
    
    manifest = om.finalize()
    if manifest.empty:
        print("AUDIT FAILED: No valid banners passed structural evaluation.")
        return
        
    manifest.to_csv("sentinel_omicron_manifest.csv", index=False)
    
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
        cv2.imwrite(os.path.join(OUT_DIR, f"om_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_omicron()