import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_iota_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 200

# KINEMATIC & CHROMA LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 12 
MIN_V, MAX_V = 4.0, 18.0
MIN_FILL_RATE = 0.55 # Replaces the undefined RELAXED_FILL
MIN_BANNER_WIDTH = 500

class SiblingClusterIota:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops) 
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.v_history = []
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self._record(frame_idx, is_init=True)

    def _record(self, f, is_init=False):
        """Captures full spatial footprint for the manifest."""
        for idx, t in enumerate(self.tops):
            self.history.append({
                "frame": f, 
                "id": self.id, 
                "sibling_idx": idx, # 0=Top banner, 1=Bottom banner
                "y_top": t,
                "y_bottom": t + BANNER_H,
                "x_start": 40, 
                "x_end": 1240, 
                "v": self.v, 
                "valid": self.is_validated,
                "is_inertial": is_init
            })

    def update(self, new_tops, f):
        self.age += 1
        is_locked = self.is_validated and (self.tops[0] < LOCK_ZONE_Y)
        target = self.tops[0] - self.v
        matches = [t for t in new_tops if abs(t - target) < 15]
        
        visual_match = False
        if is_locked:
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                if abs(best - target) < 3.0: # Gated Micro-Snap
                    self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
                    visual_match = True
            if not visual_match:
                self.tops = [t - self.v for t in self.tops]
        else:
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                if MIN_V <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    self.v = (self.v * 0.7) + (actual_v * 0.3)
                    self.v_history.append(actual_v)
                    if len(self.v_history) > 5: self.v_history.pop(0)
                    self.tops = [best] + [best + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
                    visual_match = True
            
            if not visual_match:
                self.tops = [t - self.v for t in self.tops]
                self.consistency_score = max(0, self.consistency_score - 1)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            if self.v_history: self.v = sum(self.v_history) / len(self.v_history)
            for item in self.history: item['valid'] = True

        self._record(f, is_init=not visual_match)
        if self.age > 40 and self.consistency_score < 2 and not is_locked: self.active = False
        if self.tops[0] < -50: self.active = False
        return self.active

class SentinelIota:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def check_chroma_and_structure(self, img_bgr, t):
        """Filters out high-red damage numbers."""
        h, w, _ = img_bgr.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        
        row_bgr = img_bgr[y_probe, int(w*0.1):int(w*0.9)]
        row_gray = cv2.cvtColor(row_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
        
        # 1. Fill Rate
        fill_rate = np.mean(row_gray < 75)
        if fill_rate < MIN_FILL_RATE: return False
        
        # 2. Chroma Check (R vs G balance)
        r_avg = np.mean(row_bgr[:, 2])
        g_avg = np.mean(row_bgr[:, 1])
        if r_avg > (g_avg + 30): return False # Too red = Damage Number
        
        return True

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_chroma_and_structure(img_bgr, t)]
        
        for c in self.clusters:
            if not c.update(valid_tops, f_idx):
                self.master_history.extend(c.history)
                
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 50 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterIota(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]

    def finalize_manifest(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty: return df
        # Pruning: Only validated and non-inertial frames
        df = df[df['valid'] & ~df['is_inertial']]
        # Sort and clean
        df = df.sort_values(by=['frame', 'id', 'sibling_idx']).drop_duplicates(subset=['frame', 'id', 'sibling_idx'])
        return df

def run_sentinel_iota():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    iota = SentinelIota()
    
    print("PHASE 1: Scanning frames...")
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: iota.process_frame(img_bgr, i)
    
    manifest = iota.finalize_manifest()
    manifest.to_csv("sentinel_iota_manifest.csv", index=False)
    print(f"Manifest complete. {len(manifest)} valid data points found.")
    
    print("PHASE 2: Generating 'Pristine' Overlays...")
    for i in range(START_F, min(END_F, len(all_files))):
        frame_data = manifest[manifest['frame'] == i]
        if frame_data.empty: continue
        
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        overlay = img_bgr.copy()
        for _, row in frame_data.iterrows():
            y_draw = int(row['y_top'] + DRAW_OFFSET)
            cv2.rectangle(overlay, (int(row['x_start']), y_draw), (int(row['x_end']), y_draw + BANNER_H), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"iota_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_iota()