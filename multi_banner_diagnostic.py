import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_omicron_debug"
START_F, END_F = 1980, 2080 

# GEOMETRY & AI OFFSET
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
# Adjusted offset to move the mask UP to better cover the text
DRAW_OFFSET = -12 

# KINEMATIC LIMITS
EXPECTED_V = 10.0
MIN_V_TO_STAY_ALIVE = 2.5 
MAX_SPACING_DRIFT = 6.0 # Sibling banners must move together

class SiblingClusterOmicron:
    def __init__(self, tops):
        self.tops = sorted(tops) # [y_top1, y_top2]
        self.v = EXPECTED_V
        self.age = 0
        self.v_history = []
        self.locked_spacing = None
        if len(self.tops) > 1:
            self.locked_spacing = self.tops[1] - self.tops[0]
        self.active = True
        self.id = np.random.randint(1000, 9999)

    def update(self, new_tops):
        self.age += 1
        
        # 1. PREDICTED POSITION
        target_leader = self.tops[0] - self.v
        
        # 2. MATCHING WITH RIGIDITY
        # Find candidates near predicted leader
        matches = [t for t in new_tops if abs(t - target_leader) < 25]
        
        if matches:
            best_leader = min(matches, key=lambda t: abs(t - target_leader))
            measured_v = self.tops[0] - best_leader
            
            # Smooth velocity
            self.v = (self.v * 0.7) + (measured_v * 0.3)
            self.v_history.append(self.v)
            if len(self.v_history) > 10: self.v_history.pop(0)
            
            # Update Tops: If we have siblings, keep them rigidly spaced
            if self.locked_spacing:
                self.tops = [best_leader, best_leader + self.locked_spacing]
            else:
                self.tops = [best_leader]
            
            self.lost_count = 0
        else:
            # COAST
            self.tops = [t - self.v for t in self.tops]
            self.lost_count = self.age # Using age to track lost frames in coast
        
        # 3. VELOCITY TREND GATING (The "HUD Glue" Fix)
        # If we have enough history and the banner stops moving, it's noise/HUD
        if len(self.v_history) >= 5:
            avg_v = sum(self.v_history[-5:]) / 5
            if avg_v < MIN_V_TO_STAY_ALIVE and self.tops[0] < 150:
                self.active = False # Kill track if it stalls near the HUD
                
        # 4. EXIT CONDITION
        if (self.tops[0] + BANNER_H) < SCAN_Y_START or self.age > 120:
            self.active = False
            
        return self.active

class SentinelOmicron:
    def __init__(self):
        self.clusters = []

    def get_clean_edges(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        strip = img_gray[:, c1:c2]
        ints = np.mean(strip, axis=1)
        
        # Edge Detection
        grad = np.diff(ints.astype(float))
        # We look for a sharp DROP (Light -> Dark)
        tops = np.where(grad < -7.0)[0]
        
        valid_tops = []
        for t in tops:
            if not (SCAN_Y_START <= t <= SCAN_Y_END): continue
            # Neighborhood check: Is it brighter above than below?
            if t > 5 and t + 10 < h:
                if np.mean(ints[t-5:t]) > np.mean(ints[t:t+10]) + 10:
                    valid_tops.append(t)
        return valid_tops, ints

    def process_frame(self, img_gray):
        tops, intensities = self.get_clean_edges(img_gray)
        
        # 1. Update existing
        for c in self.clusters:
            c.update(tops)
            
        # 2. Spatial Pruning (Merge Overlapping Clusters)
        self.clusters.sort(key=lambda x: x.tops[0])
        for i in range(len(self.clusters)-1):
            if abs(self.clusters[i].tops[0] - self.clusters[i+1].tops[0]) < 20:
                self.clusters[i+1].active = False # Prune the younger/duplicate
        
        # 3. Birth Logic (Nucleation Zone > 250)
        birth_tops = [t for t in tops if t > 250]
        # Only spawn if no active tracker is nearby
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 40 for c in self.clusters for t in c.tops)]
        
        if len(birth_tops) >= 2:
            # Handle potential multi-banner arrival
            if abs(birth_tops[0] - birth_tops[1]) < 100:
                self.clusters.append(SiblingClusterOmicron([birth_tops[0], birth_tops[1]]))
        elif len(birth_tops) == 1:
            self.clusters.append(SiblingClusterOmicron([birth_tops[0]]))

        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters, intensities

def run_sentinel_omicron():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    so = SentinelOmicron()
    manifest = []
    
    for i in range(START_F, END_F):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters, ints = so.process_frame(img_gray)
        
        overlay = img_bgr.copy()
        for c in clusters:
            # Kinematic Filter: Only draw if it has moved from its birth
            if c.age > 5:
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    cv2.rectangle(overlay, (0, y_draw), (img_bgr.shape[1], y_draw + BANNER_H), (0, 0, 255), -1)
                    manifest.append({"frame": i, "id": c.id, "y_top": t, "v": c.v})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        status = f"CLUSTERS: {len(clusters)}"
        cv2.putText(img_bgr, f"OMICRON F:{i} | {status}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"omicron_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_omicron_manifest.csv", index=False)
    print("Sentinel Omicron Lab complete.")

if __name__ == "__main__":
    run_sentinel_omicron()