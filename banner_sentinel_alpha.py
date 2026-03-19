import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_gamma_debug"
START_F, END_F = 2000, 4000 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 

# KINEMATIC LAWS
EXPECTED_V = 10.0
CONSISTENCY_WINDOW = 6 # Faster validation to catch nucleation
MIN_V, MAX_V = 6.0, 16.0
NUCLEATION_ZONE = 250

class SiblingClusterGamma:
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
        
        # 1. PREDICTED POSITION (Inertia)
        target = self.tops[0] - self.v
        
        # 2. GATED SNAPPING
        # Look for a visual match near the prediction
        matches = [t for t in new_tops if abs(t - target) < 15]
        
        best_match = None
        if matches:
            # Pick match that most closely aligns with our CURRENT velocity
            best_match = min(matches, key=lambda t: abs((self.tops[0] - t) - self.v))
            measured_v = self.tops[0] - best_match
            
            # Validation logic: Is the movement "Banner-like"?
            if MIN_V <= measured_v <= MAX_V:
                self.consistency_score += 1
                # Velocity Smoothing: 50% physics / 50% observation
                self.v = (self.v * 0.5) + (measured_v * 0.5)
                # Update positions
                self.tops = [best_match] + [best_match + (self.tops[i]-self.tops[0]) for i in range(1, len(self.tops))]
            else:
                self.consistency_score = max(0, self.consistency_score - 1)
        else:
            # COASTING (Inertia only)
            self.tops = [t - self.v for t in self.tops]
            self.consistency_score = max(0, self.consistency_score - 1)

        # Trigger Validation (Backfill)
        if not self.is_validated and self.consistency_score >= CONSISTENCY_WINDOW:
            self.is_validated = True
            for item in self.history: item['valid'] = True

        self._record(f)

        # KILL: If it stalls or drifts out
        if self.age > 40 and self.consistency_score < 3: self.active = False
        if self.tops[0] < -50: self.active = False
        return self.active

class SentinelGamma:
    def __init__(self):
        self.clusters = []
        self.final_manifest = []

    def get_structural_score(self, img_gray, t):
        """Measures texture ratio to distinguish text from grid rows."""
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        if t + 42 >= h: return 0.0
        
        # Center row (where text should be)
        core = img_gray[int(t + 22), c1:c2].astype(float)
        # Bottom row (where padding should be)
        base = img_gray[int(t + 40), c1:c2].astype(float)
        
        core_var = np.var(core)
        base_var = np.var(base) + 1.0 # Avoid div by zero
        
        # Ratio: Text should have significantly more texture than the black base
        score = core_var / base_var
        return score

    def process_frame(self, img_gray, f_idx):
        w = img_gray.shape[1]
        center_ints = np.mean(img_gray[:, int(w*0.4):int(w*0.6)], axis=1)
        grad = np.diff(center_ints.astype(float))
        tops = np.where(grad < -7.0)[0]
        
        # Refined Candidate List
        valid_candidates = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            score = self.get_structural_score(img_gray, t)
            # Accept if it's dark and has any significant texture difference
            if score > 2.0 and np.mean(img_gray[t+10:t+35, int(w*0.4):int(w*0.6)]) < 75:
                valid_candidates.append({'y': t, 'score': score})
        
        # 1. Update existing tracks
        current_tops = [c['y'] for c in valid_candidates]
        for c in self.clusters:
            was_v = c.is_validated
            if c.update(current_tops, f_idx):
                if c.is_validated and not was_v: self.final_manifest.extend(c.history)
                elif c.is_validated: self.final_manifest.append(c.history[-1])
        
        # 2. Nucleation
        births = [c for c in valid_candidates if c['y'] > NUCLEATION_ZONE]
        for b in births:
            if not any(abs(b['y'] - t) < 45 for cl in self.clusters for t in cl.tops):
                self.clusters.append(SiblingClusterGamma([b['y']], f_idx))
        
        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_gamma():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sg = SentinelGamma()
    
    for i in range(START_F, min(END_F, len(all_files))):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = sg.process_frame(img_gray, i)
        
        overlay = img_bgr.copy()
        for c in clusters:
            # We draw BOTH Red and Yellow now so you can see the signal strength
            color = (0, 0, 255) if c.is_validated else (0, 255, 255)
            for t in c.tops:
                y_draw = int(t + DRAW_OFFSET)
                cv2.rectangle(overlay, (0, max(0, y_draw)), (img_bgr.shape[1], max(0, y_draw + BANNER_H)), color, -1)
                # Telemetry
                label = f"ID:{c.id} V:{c.v:.1f}"
                cv2.putText(img_bgr, label, (10, y_draw - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"GAMMA F:{i} | CLUSTERS: {len(clusters)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"gamma_{i:05}.png"), img_bgr)
    
    pd.DataFrame(sg.final_manifest).to_csv("sentinel_gamma_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_gamma()