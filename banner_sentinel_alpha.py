import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_delta_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 

# KINEMATIC LAWS (The 'Elastic' Gate)
EXPECTED_V = 10.0
CONSISTENCY_WINDOW = 4 
MIN_V, MAX_V = 4.0, 20.0 # Broad window to handle nucleation and jitters
NUCLEATION_ZONE = 250

class SiblingClusterDelta:
    def __init__(self, tops, frame_idx):
        self.tops = sorted(tops)
        self.v = EXPECTED_V
        self.age = 0
        self.consistency_score = 0
        self.is_validated = False 
        self.active = True
        self.id = np.random.randint(1000, 9999)
        self.history = []
        self._record(frame_idx, 0.0)

    def _record(self, f, score):
        for t in self.tops:
            self.history.append({
                "frame": f, "id": self.id, "y": t, "v": self.v, 
                "score": score, "valid": self.is_validated
            })

    def update(self, valid_cands, f):
        self.age += 1
        target = self.tops[0] - self.v
        
        # 1. GATED SNAPPING
        # Look for the best match near the predicted position
        best_match = None
        best_score = 0.0
        
        for cand in valid_cands:
            if abs(cand['y'] - target) < 25:
                actual_v = self.tops[0] - cand['y']
                if MIN_V <= actual_v <= MAX_V:
                    best_match = cand['y']
                    best_score = cand['score']
                    break

        if best_match is not None:
            measured_v = self.tops[0] - best_match
            self.consistency_score += 1
            # Weighted velocity update
            self.v = (self.v * 0.5) + (measured_v * 0.5)
            self.tops = [best_match] + [best_match + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
        else:
            # COASTING (Inertia)
            self.tops = [t - self.v for t in self.tops]
            if self.tops[0] > 150: # Only penalize misses in the clear grid area
                self.consistency_score = max(0, self.consistency_score - 1)

        # TRIGGER VALIDATION (Sticky)
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f, best_score)

        # KILL: If it completely stalls or drifts off
        if self.age > 40 and self.consistency_score < 2 and not self.is_validated: 
            self.active = False
        if self.tops[0] < -40: 
            self.active = False
            
        return self.active

class SentinelDelta:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def get_structural_score(self, img_gray, t):
        """Measures texture ratio (Text-Core vs Black-Base)."""
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        if t + 42 >= h: return 0.0
        
        # Sample the center (Text) and base (Black)
        core_var = np.var(img_gray[int(t + 22), c1:c2].astype(float))
        base_var = np.var(img_gray[int(t + 40), c1:c2].astype(float)) + 1.0
        
        return core_var / base_var

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -7.0)[0]
        
        valid_cands = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            score = self.get_structural_score(img_gray, t)
            # RELAXED RATIO: 1.3 instead of 2.0
            if score > 1.3:
                valid_cands.append({'y': t, 'score': score})
        
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(valid_cands, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # Nucleation
        births = [c for c in valid_cands if c['y'] > NUCLEATION_ZONE]
        for b in births:
            if not any(abs(b['y'] - t) < 45 for cl in self.clusters for t in cl.tops):
                self.clusters.append(SiblingClusterDelta([b['y']], f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_delta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sd = SentinelDelta()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = sd.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            color = (0, 0, 255) if c.is_validated else (0, 255, 255)
            for t in c.tops:
                y_draw = int(t + DRAW_OFFSET)
                cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), color, -1)
                # TELEMETRY LABELS
                label = f"ID:{c.id} V:{c.v:.1f} R:{c.consistency_score}"
                cv2.putText(img_bgr, label, (10, y_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"DELTA F:{i} | VALIDATED: {len([c for c in clusters if c.is_validated])}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"delta_{i:05}.png"), img_bgr)
    
    pd.DataFrame(sd.final_manifest).to_csv("sentinel_delta_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_delta()