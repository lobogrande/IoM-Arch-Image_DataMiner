import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_pi_debug"
START_F, END_F = 1980, 2080 

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H = 45
DRAW_OFFSET = -12 
ICON_SAFETY_MARGIN = 15 # Tracking persists until icon clears HUD

# KINEMATIC LIMITS
EXPECTED_V = 10.0
MIN_V_TO_STAY_ALIVE = 2.0 
NUCLEATION_ZONE = 250

class SiblingClusterPi:
    def __init__(self, tops):
        self.tops = sorted(tops)
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
        target_leader = self.tops[0] - self.v
        matches = [t for t in new_tops if abs(t - target_leader) < 25]
        
        if matches:
            best_leader = min(matches, key=lambda t: abs(t - target_leader))
            measured_v = self.tops[0] - best_leader
            self.v = (self.v * 0.7) + (measured_v * 0.3)
            self.v_history.append(self.v)
            if len(self.v_history) > 10: self.v_history.pop(0)
            
            if self.locked_spacing:
                self.tops = [best_leader, best_leader + self.locked_spacing]
            else:
                self.tops = [best_leader]
        else:
            # COASTING: Use last known velocity
            self.tops = [t - self.v for t in self.tops]
        
        # --- IMPROVED EXIT LOGIC ---
        # Calculate the "Dirty Bottom" (Bottom edge + dangling icon)
        # We track the last sibling in the cluster for the bottom-most point
        dirty_bottom = self.tops[-1] + BANNER_H + ICON_SAFETY_MARGIN
        
        # Kill only if the dirty bottom clears the scan start
        if dirty_bottom < SCAN_Y_START:
            self.active = False
            
        # Velocity Stall Check (HUD Glue prevention)
        if len(self.v_history) >= 5:
            if (sum(self.v_history[-5:]) / 5) < MIN_V_TO_STAY_ALIVE and self.tops[0] < 120:
                self.active = False
                
        return self.active

class SentinelPi:
    def __init__(self):
        self.clusters = []

    def get_clean_edges(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        strip = img_gray[:, c1:c2]
        ints = np.mean(strip, axis=1)
        grad = np.diff(ints.astype(float))
        tops = np.where(grad < -7.0)[0]
        
        valid_tops = []
        for t in tops:
            if not (SCAN_Y_START - 20 <= t <= SCAN_Y_END + 20): continue
            if t > 5 and t + 10 < h:
                if np.mean(ints[t-5:t]) > np.mean(ints[t:t+10]) + 10:
                    valid_tops.append(t)
        return valid_tops

    def process_frame(self, img_gray):
        tops = self.get_clean_edges(img_gray)
        for c in self.clusters:
            c.update(tops)
            
        # Merge duplicates
        self.clusters.sort(key=lambda x: x.tops[0])
        for i in range(len(self.clusters)-1):
            if abs(self.clusters[i].tops[0] - self.clusters[i+1].tops[0]) < 20:
                self.clusters[i+1].active = False
        
        # New cluster detection
        birth_tops = [t for t in tops if t > NUCLEATION_ZONE]
        birth_tops = [bt for bt in birth_tops if not any(abs(bt - t) < 40 for c in self.clusters for t in c.tops)]
        
        if len(birth_tops) >= 2:
            self.clusters.append(SiblingClusterPi(birth_tops[:2]))
        elif len(birth_tops) == 1:
            self.clusters.append(SiblingClusterPi([birth_tops[0]]))

        self.clusters = [c for c in self.clusters if c.active]
        return self.clusters

def run_sentinel_pi():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    sp = SentinelPi()
    
    for i in range(START_F, END_F):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters = sp.process_frame(img_gray)
        
        overlay = img_bgr.copy()
        for c in clusters:
            # --- TWEAK: LOWERED DRAWING THRESHOLD ---
            # If it's in the birth zone, trust it immediately (age > 1)
            # This fixes the 6-frame nucleation lag.
            if c.age > 1:
                for t in c.tops:
                    y_draw = int(t + DRAW_OFFSET)
                    y1, y2 = max(0, y_draw), max(0, y_draw + BANNER_H)
                    cv2.rectangle(overlay, (0, y1), (img_bgr.shape[1], y2), (0, 0, 255), -1)
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"PI F:{i} | CLUSTERS: {len(clusters)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"pi_{i:05}.png"), img_bgr)

if __name__ == "__main__":
    run_sentinel_pi()