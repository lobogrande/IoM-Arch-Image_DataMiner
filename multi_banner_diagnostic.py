import cv2
import numpy as np
import os
import pandas as pd

# --- CALIBRATED CONSTANTS ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_mu_debug"
# Target frames for the double event identified by user
START_F, END_F = 2010, 2070 

SCAN_Y_START, SCAN_Y_END = 40, 450
INTENSITY_THRESH = 40.0
HUC_VAR_THRESH = 500.0  # Allow slightly more noise during nucleation
GLOBAL_VELOCITY = 10.0

class SiblingCluster:
    def __init__(self, centers):
        self.centers = sorted(centers) # [y_top_banner, y_bot_banner]
        self.age = 0
        self.v = GLOBAL_VELOCITY
        self.locked_spacing = None
        self.active = True
        self.id = np.random.randint(1000, 9999)

    def update(self, new_candidates):
        self.age += 1
        # Predict where the centroid should be
        centroid = sum(self.centers) / len(self.centers)
        target_centroid = centroid - self.v
        
        # 1. EXPANSION PHASE (Age < 4)
        if self.age < 4:
            # Look for ANY two candidates close to the last centroid
            # that are "separating"
            best_pair = None
            min_dist = 50
            for i in range(len(new_candidates)):
                for j in range(i + 1, len(new_candidates)):
                    pair_centroid = (new_candidates[i] + new_candidates[j]) / 2
                    if abs(pair_centroid - target_centroid) < min_dist:
                        best_pair = [new_candidates[i], new_candidates[j]]
                        min_dist = abs(pair_centroid - target_centroid)
            
            if best_pair:
                self.centers = sorted(best_pair)
                return True
        
        # 2. SYNC PHASE (Age >= 4)
        if self.locked_spacing is None:
            self.locked_spacing = self.centers[1] - self.centers[0]
            
        # Move the cluster as a single unit
        target_leader = self.centers[0] - self.v
        best_leader = None
        min_err = 25
        for cand in new_candidates:
            if abs(cand - target_leader) < min_err:
                best_leader = cand
                min_err = abs(cand - target_leader)
        
        if best_leader:
            measured_v = self.centers[0] - best_leader
            self.v = (self.v * 0.8) + (measured_v * 0.2)
            self.centers = [best_leader, best_leader + self.locked_spacing]
            return True
            
        return False # Lost tracking

class SentinelMu:
    def __init__(self):
        self.clusters = []

    def get_raw_candidates(self, img_gray):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.2), int(w * 0.8)
        ints = np.mean(img_gray[:, c1:c2], axis=1)
        vars = np.var(img_gray[:, c1:c2], axis=1)
        
        mask = ((ints < INTENSITY_THRESH) & (vars < HUC_VAR_THRESH)).astype(np.uint8)
        # Dilate slightly to connect fragmented text rows
        mask = cv2.dilate(mask, np.ones((5, 1), np.uint8))
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cands = []
        for i in range(1, num_labels):
            y_top, height = stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_HEIGHT]
            # Banners start narrow, so we accept heights from 15 to 80
            if 15 <= height <= 80 and SCAN_Y_START <= y_top <= SCAN_Y_END:
                cands.append(y_top + height // 2)
        return cands, ints

    def run_frame(self, img_gray):
        cands, intensities = self.get_raw_candidates(img_gray)
        
        # Update existing clusters
        for cluster in self.clusters:
            if not cluster.update(cands):
                cluster.active = False
        
        # Check for NEW clusters (Nucleation)
        # If we see 2+ candidates near each other in the Birth Zone (>250)
        birth_zone_cands = [c for c in cands if c > 250]
        if len(birth_zone_cands) >= 2:
            # If they aren't already tracked
            if not any(abs(sum(cluster.centers)/2 - sum(birth_zone_cands[:2])/2) < 40 for cluster in self.clusters):
                self.clusters.append(SiblingCluster(birth_zone_cands[:2]))
        
        self.clusters = [c for c in self.clusters if c.active and c.centers[0] > SCAN_Y_START]
        return self.clusters, intensities

def run_sentinel_mu():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    mu = SentinelMu()
    manifest = []
    
    for i in range(START_F, END_F):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, all_files[i]))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clusters, ints = mu.run_frame(img_gray)
        
        overlay = img_bgr.copy()
        for c in clusters:
            for y_center in c.centers:
                y1, y2 = int(y_center - 22), int(y_center + 22)
                cv2.rectangle(overlay, (0, y1), (img_bgr.shape[1], y2), (0, 0, 255), -1)
                manifest.append({"frame": i, "cluster_id": c.id, "y_center": y_center, "age": c.age})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        status = f"CLUSTERS: {len(clusters)}"
        cv2.putText(img_bgr, f"MU F:{i} | {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"mu_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_mu_manifest.csv", index=False)

if __name__ == "__main__":
    run_sentinel_mu()