import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_nu_debug"
START_F, END_F = 1980, 2080 # Targeted Training Lab Range

# GEOMETRY
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H_TARGET = 45

# THRESHOLDS
INTENSITY_THRESH = 42.0 # Slightly relaxed for nucleation
HUC_VAR_THRESH = 480.0
GLOBAL_VELOCITY = 10.0
VELOCITY_SMOOTHING = 0.90 

class SiblingClusterNu:
    def __init__(self, centers):
        self.centers = sorted(centers) # [y_top_banner, y_bot_banner]
        self.v = GLOBAL_VELOCITY
        self.age = 0
        self.locked_spacing = None
        self.active = True
        self.id = np.random.randint(10000, 99999)

    def update(self, new_cands):
        self.age += 1
        
        # Phase 1: SEPARATION (Age < 6)
        # We allow children to move independently to reach their final slots.
        if self.age < 6:
            new_positions = []
            for last_y in self.centers:
                # Search +/- 40 pixels for each child
                matches = [c for c in new_cands if abs(c - last_y) < 40]
                if matches:
                    # Pick match closest to expected trajectory (Up 10px)
                    best = min(matches, key=lambda c: abs(c - (last_y - 10)))
                    new_positions.append(best)
            
            if len(new_positions) >= 2:
                self.centers = sorted(new_positions)
                return True
            else:
                # Predictive coasting if one signal is lost
                self.centers = [y - 10 for y in self.centers]
                return True

        # Phase 2: LOCKED SYNC (Age >= 6)
        # Banners are now full-width and moving together.
        if self.locked_spacing is None:
            self.locked_spacing = [self.centers[i] - self.centers[0] for i in range(1, len(self.centers))]

        # Track the cluster leader and snap siblings to the relative locked spacing
        target_leader = self.centers[0] - self.v
        matches = [c for c in new_cands if abs(c - target_leader) < 25]
        
        if matches:
            best_leader = min(matches, key=lambda c: abs(c - target_leader))
            measured_v = self.centers[0] - best_leader
            self.v = (self.v * 0.9) + (measured_v * 0.1)
            self.centers = [best_leader] + [best_leader + s for s in self.locked_spacing]
            return True
        else:
            # Full cluster coasting
            self.centers = [y - self.v for y in self.centers]
            return True

class SentinelNu:
    def __init__(self):
        self.clusters = []

    def get_raw_candidates(self, img_gray):
        h, w = img_gray.shape
        # Use localized 30% center strip to find incipient narrow banners
        c1, c2 = int(w * 0.35), int(w * 0.65)
        ints = np.mean(img_gray[:, c1:c2], axis=1)
        vars = np.var(img_gray[:, c1:c2], axis=1)
        
        # HUC: Darkness + Uniformity
        mask = ((ints < INTENSITY_THRESH) & (vars < HUC_VAR_THRESH)).astype(np.uint8)
        
        # Clean small noise
        mask = cv2.dilate(mask, np.ones((5, 1), np.uint8))
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cands = []
        for i in range(1, num_labels):
            y_top, height = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
            # Height range for early vs mature banners
            if 12 <= height <= 85 and SCAN_Y_START <= y_top <= SCAN_Y_END:
                cands.append(y_top + height // 2)
        return cands, ints

    def process_frame(self, img_gray):
        cands, intensities = self.get_raw_candidates(img_gray)
        
        # 1. Update existing tracks
        for cluster in self.clusters:
            cluster.active = cluster.update(cands)
            
        # 2. Birth Detection
        # Identify new candidates not currently assigned to a cluster
        potential_new = []
        for cand in cands:
            if not any(any(abs(cand - yc) < 30 for yc in cl.centers) for cl in self.clusters):
                potential_new.append(cand)
        
        if len(potential_new) >= 2:
            potential_new.sort()
            for i in range(len(potential_new)-1):
                if potential_new[i] > 200: # Search Birth Zone
                    self.clusters.append(SiblingClusterNu([potential_new[i], potential_new[i+1]]))
                    break

        self.clusters = [c for c in self.clusters if c.active and all(y > 0 for y in c.centers)]
        return self.clusters, intensities

def run_sentinel_nu():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    sn = SentinelNu()
    manifest = []
    
    for i in range(START_F, END_F):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters, ints = sn.process_frame(img_gray)
        
        overlay = img_bgr.copy()
        for cluster in clusters:
            for y_center in cluster.centers:
                y1, y2 = int(y_center - 22), int(y_center + 22)
                cv2.rectangle(overlay, (0, y1), (img_bgr.shape[1], y2), (0, 0, 255), -1)
                manifest.append({"frame": i, "id": cluster.id, "y": y_center})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"NU F:{i} | CLUSTERS: {len(clusters)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"nu_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_nu_manifest.csv", index=False)
    print("Sentinel Nu Lab complete. Check 'sentinel_nu_manifest.csv'.")

if __name__ == "__main__":
    run_sentinel_nu()