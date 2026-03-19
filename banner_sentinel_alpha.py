import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_zeta_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
HUD_DANGER_ZONE = 145

# KINEMATIC & SIGNAL LAWS
EXPECTED_V = 10.1
CONSISTENCY_WINDOW = 10 # Slightly faster validation
MIN_V, MAX_V = 5.0, 18.0
NUCLEATION_ZONE = 250
RELAXED_FILL = 0.55 # 55% instead of 85%

class SiblingClusterZeta:
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
        
        matches = [t for t in new_tops if abs(t - target) < 30]
        best_match = None
        
        if matches:
            temp_best = min(matches, key=lambda t: abs(t - target))
            actual_v = self.tops[0] - temp_best
            
            # KINEMATIC GATE: Only move UP
            if 3.0 <= actual_v <= MAX_V:
                best_match = temp_best
                self.v = (self.v * 0.7) + (actual_v * 0.3)
                self.consistency_score += 1
            else:
                self.consistency_score = max(0, self.consistency_score - 1)

        if best_match is not None:
            self.tops = [best_match] + [best_match + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            # COASTING
            self.tops = [t - self.v for t in self.tops]
            if self.tops[0] > HUD_DANGER_ZONE:
                self.consistency_score = max(0, self.consistency_score - 2)

        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        if self.age > 30 and self.consistency_score < 2: self.active = False
        if self.tops[0] < -50: self.active = False
        return self.active

class SentinelZeta:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def check_ensemble(self, img_gray, t):
        """Combines horizontal fill with internal text variance."""
        h, w = img_gray.shape
        y_probe = int(t + 15)
        if y_probe >= h: return False
        
        # 1. HORIZONTAL FILL (55%)
        x_start, x_end = int(w * 0.1), int(w * 0.9)
        row_strip = img_gray[y_probe, x_start:x_end]
        fill_rate = np.mean(row_strip < 75)
        
        # 2. TEXT VARIANCE (Confirmation)
        # Banners have text; grid rows and floor changes do not.
        variance = np.var(row_strip.astype(float))
        
        return (fill_rate > RELAXED_FILL) and (variance > 10.0)

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -8.0)[0]
        
        valid_tops = [t for t in tops if SCAN_Y_START <= t <= SCAN_Y_END and self.check_ensemble(img_gray, t)]
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Birth
        birth = [t for t in valid_tops if t > NUCLEATION_ZONE]
        birth = [bt for bt in birth if not any(abs(bt - t) < 50 for c in self.clusters for t in c.tops)]
        if birth: self.clusters.append(SiblingClusterZeta(birth, f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_zeta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sz = SentinelZeta()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = sz.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            if c.is_validated:
                # Solid Red for confirmed banners
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), (0, 0, 255), -1)
            else:
                # Hollow Yellow for candidates (probation)
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(img_bgr, (20, max(0, y_draw)), (img_bgr.shape[1]-20, max(0, y_draw + BANNER_H)), (0, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"ZETA F:{i} | CLUSTERS: {len(clusters)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"zeta_{i:05}.png"), img_bgr)
    
    pd.DataFrame(sz.final_manifest).to_csv("sentinel_zeta_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_zeta()