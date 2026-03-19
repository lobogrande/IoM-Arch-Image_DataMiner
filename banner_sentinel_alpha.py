import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_lambda_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START = 40
SCAN_Y_END = 450
BANNER_H = 45
DRAW_OFFSET = -12 
LOCK_ZONE_Y = 200 # Transition to Ballistic Lock
NUCLEATION_Y_START = 350 # Banners must be in this zone to be "born"
NUCLEATION_Y_END = 385

# KINEMATIC LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 10 
MIN_V, MAX_V = 6.0, 16.0
MIN_VALID_DISPLACEMENT = 60.0 

class SiblingClusterLambda:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.dist_moved = 0.0
        self.v_samples = [] # Used for "Golden Mean" calculation
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self.mode = "N" # N=Nucleation, C=Cruising, B=Ballistic
        self._record(frame_idx)

    def _record(self, f):
        for idx, t in enumerate(self.tops):
            self.history.append({
                "frame": f, "id": self.id, "sibling_idx": idx,
                "y_top": float(t), "v": self.v, 
                "mode": self.mode, "valid": self.is_validated
            })

    def update(self, new_tops, f):
        self.age += 1
        
        # 1. KINEMATIC MODE SELECTION
        if self.is_validated and self.tops[0] < LOCK_ZONE_Y:
            self.mode = "B" # Ballistic Lock
        elif self.is_validated:
            self.mode = "C" # Cruising
        else:
            self.mode = "N" # Nucleation/Vetting

        target = self.tops[0] - self.v
        matches = [t for t in new_tops if abs(t - target) < 20]
        
        visual_match = False
        if self.mode == "B":
            # BALLISTIC MODE: Ignore screen noise, allow only tiny Micro-Snaps
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                if abs(best - target) < 2.0: # Very tight gate
                    self.tops = [best] + [best + (self.tops[idx]-self.tops[0]) for idx in range(1, len(self.tops))]
                    visual_match = True
            if not visual_match:
                self.tops = [t - self.v for t in self.tops]
        else:
            # NORMAL TRACKING
            if matches:
                best = min(matches, key=lambda t: abs(t - target))
                actual_v = self.tops[0] - best
                if MIN_V <= actual_v <= MAX_V:
                    self.consistency_score += 1
                    self.dist_moved += actual_v
                    # Velocity Smoothing
                    self.v = (self.v * 0.7) + (actual_v * 0.3)
                    # Collect samples for Golden Mean (Cruising Phase)
                    if self.tops[0] < 350: self.v_samples.append(actual_v)
                    
                    self.tops = [best] + [best + (self.tops[idx]-self.tops[0]) for idx in range(1, len(self.tops))]
                    visual_match = True
            
            if not visual_match:
                self.tops = [t - self.v for t in self.tops]
                self.consistency_score = max(0, self.consistency_score - 1)

        # 2. VALIDATION TRIGGER
        if not self.is_validated:
            if self.consistency_score >= CONSISTENCY_WINDOW and self.dist_moved >= MIN_VALID_DISPLACEMENT:
                self.is_validated = True
                # CALIBRATE GOLDEN MEAN: Use all cruising samples for the ballistic lock
                if self.v_samples: self.v = sum(self.v_samples) / len(self.v_samples)
                for item in self.history: item['valid'] = True

        self._record(f)
        
        # TERMINATION: Bottom edge must clear the HUD (y < 40)
        if (self.tops[0] + BANNER_H) < 30: self.active = False
        if self.age > 40 and not self.is_validated: self.active = False
        return self.active

class SentinelLambda:
    def __init__(self):
        self.clusters = []
        self.master_history = []

    def check_ensemble(self, img_bgr, t, age):
        h, w, _ = img_bgr.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        
        row_bgr = img_bgr[y_probe, int(w*0.1):int(w*0.9)]
        row_gray = cv2.cvtColor(row_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).flatten()
        
        # 1. Structural Check
        fill_rate = np.mean(row_gray < 75)
        variance = np.var(row_gray.astype(float))
        
        # Strict nucleation check (must have text signature)
        if age < 8 and variance < 15.0: return False
        return (fill_rate > 0.55) and (variance > 10.0)

    def process_frame(self, img_bgr, f_idx):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        for c in self.clusters:
            # Feed current age to the structure check
            valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_ensemble(img_bgr, t, c.age)]
            if not c.update(valid_tops, f_idx):
                self.master_history.extend(c.history)
                
        # BIRTH LOGIC (Tightened Nucleation Zone)
        birth_tops = [t for t in tops if NUCLEATION_Y_START <= t <= NUCLEATION_Y_END]
        birth_tops = [t for t in birth_tops if self.check_ensemble(img_bgr, t, 0)]
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 60 for c in self.clusters for t in c.tops)]
        
        if birth_tops:
            self.clusters.append(SiblingClusterLambda(birth_tops, f_idx))
        self.clusters = [c for c in self.clusters if c.active]

    def finalize_manifest(self):
        for c in self.clusters: self.master_history.extend(c.history)
        df = pd.DataFrame(self.master_history)
        if df.empty: return df
        df = df[df['valid']].sort_values(['frame', 'id', 'sibling_idx'])
        return df.drop_duplicates(subset=['frame', 'id', 'sibling_idx'])

def run_sentinel_lambda():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sl = SentinelLambda()
    
    print("PHASE 1: Calibrating Golden Mean Velocities...")
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is not None: sl.process_frame(img_bgr, i)
    
    manifest = sl.finalize_manifest()
    manifest.to_csv("sentinel_lambda_manifest.csv", index=False)
    
    print("PHASE 2: Generating Pristine Overlays...")
    for i in range(START_F, min(END_F, len(all_files))):
        frame_data = manifest[manifest['frame'] == i]
        if frame_data.empty: continue
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        overlay = img_bgr.copy()
        for _, row in frame_data.iterrows():
            y_draw = int(row['y_top'] + DRAW_OFFSET)
            cv2.rectangle(overlay, (40, y_draw), (1240, y_draw + BANNER_H), (0, 0, 255), -1)
            label = f"ID:{int(row['id'])} [{row['mode']}] V:{row['v']:.2f}"
            cv2.putText(img_bgr, label, (50, y_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.imwrite(os.path.join(OUT_DIR, f"lam_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_lambda()