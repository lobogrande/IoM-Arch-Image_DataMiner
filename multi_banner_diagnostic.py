import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_xi_debug"
START_F, END_F = 1980, 2080 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -6 # Shift mask UP by 6 pixels to center on text

# THRESHOLDS
INTENSITY_THRESH = 42.0 
HUC_VAR_THRESH = 500.0
MIN_VALID_MOTION = 4.0 # Pixels moved before a cluster is "Valid"

class SiblingClusterXi:
    def __init__(self, tops):
        self.tops = sorted(tops) # [y_top_1, y_top_2]
        self.v = 10.0
        self.age = 0
        self.dist_moved = 0.0
        self.is_validated = False # Must move to become True
        self.active = True
        self.id = np.random.randint(1000, 9999)

    def update(self, new_tops):
        self.age += 1
        
        # 1. TRACKING LOGIC
        # Find matches for each 'top' in the cluster
        matches = []
        for last_top in self.tops:
            target = last_top - self.v
            # Look for a sharp down-edge within 20px of prediction
            found = [t for t in new_tops if abs(t - target) < 25]
            if found:
                best = min(found, key=lambda t: abs(t - target))
                matches.append(best)
        
        if len(matches) >= 1:
            # We have a lock. Update velocity and positions
            actual_v = self.tops[0] - matches[0]
            self.v = (self.v * 0.8) + (actual_v * 0.2)
            self.dist_moved += abs(actual_v)
            
            # Update all tops (maintain spacing if one is lost)
            if len(matches) == len(self.tops):
                self.tops = sorted(matches)
            else:
                # Maintain relative spacing of the lost sibling
                spacing = self.tops[1] - self.tops[0]
                self.tops = [matches[0], matches[0] + spacing]
        else:
            # COAST
            self.tops = [t - self.v for t in self.tops]
            self.dist_moved += self.v
            
        # 2. VALIDATION GATING
        # If it hasn't moved after 10 frames, it's noise/HUD. Kill it.
        if self.age > 10 and self.dist_moved < MIN_VALID_MOTION:
            self.active = False
        elif self.dist_moved >= MIN_VALID_MOTION:
            self.is_validated = True
            
        # 3. EXIT CONDITION
        if any(t < SCAN_Y_START - 20 for t in self.tops) or self.age > 100:
            self.active = False
            
        return self.active

class SentinelXi:
    def __init__(self):
        self.clusters = []

    def find_top_edges(self, img_gray):
        """Finds sharp transitions from light to dark (Top edges)."""
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        strip = img_gray[:, c1:c2]
        ints = np.mean(strip, axis=1)
        
        # Vertical Gradient (Negative = Downward transition to dark)
        grad = np.diff(ints.astype(float))
        tops = np.where(grad < -6.0)[0]
        
        valid_tops = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            # Check if it's a 'Wide' dark block (Uniformity check)
            # A banner should stay dark for at least 15 pixels below the edge
            if t + 15 < h:
                if np.mean(ints[t:t+15]) < INTENSITY_THRESH:
                    valid_tops.append(t)
        return valid_tops, ints

    def process(self, img_gray):
        tops, intensities = self.find_top_edges(img_gray)
        
        # Update existing
        for c in self.clusters:
            c.update(tops)
            
        # Spawn new
        # Look for a pair of new tops in the nucleation zone (>200)
        birth_tops = [t for t in tops if t > 200]
        # Ensure we don't double-track
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 30 for c in self.clusters for t in c.tops)]
        
        if len(birth_tops) >= 2:
            self.clusters.append(SiblingClusterXi(birth_tops[:2]))
        elif len(birth_tops) == 1:
            # Handle single banner case too
            self.clusters.append(SiblingClusterXi([birth_tops[0]]))

        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters, intensities

def run_sentinel_xi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    xi = SentinelXi()
    manifest = []
    
    for i in range(START_F, END_F):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters, ints = xi.process(img_gray)
        
        overlay = img_bgr.copy()
        for c in clusters:
            # ONLY DRAW IF VALIDATED (MOVED)
            if not c.is_validated: continue
            
            for t in c.tops:
                y_draw = int(t + DRAW_OFFSET)
                cv2.rectangle(overlay, (0, y_draw), (img_bgr.shape[1], y_draw + BANNER_H), (0, 0, 255), -1)
                manifest.append({"frame": i, "id": c.id, "y_top": t})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"XI F:{i} | CLUSTERS: {len(clusters)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"xi_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_xi_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_xi()