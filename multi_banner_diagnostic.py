import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_lambda_v2_debug"
# Targeting the user-identified double event at frame 2014
TARGET_IDX = 2014 

# --- GEOMETRIC CONSTRAINTS ---
SCAN_Y_START, SCAN_Y_END = 40, 450
BANNER_H_TARGET = 45

# --- CALIBRATED THRESHOLDS ---
INTENSITY_THRESH = 38.0  # Max average darkness
HUC_VAR_THRESH = 450.0   # Max horizontal variance (The "Uniformity Floor")
GLOBAL_VELOCITY = 10.0
VELOCITY_SMOOTHING = 0.90 

class MultiTracker:
    def __init__(self, y_top):
        self.y_top = float(y_top)
        self.v = GLOBAL_VELOCITY
        self.lost = 0
        self.active = True
        self.id = np.random.randint(100, 999)

class SentinelLambdaV2:
    def __init__(self):
        self.trackers = []

    def get_huc_candidates(self, img_gray):
        """Identifies rows that are both dark AND horizontally uniform."""
        h, w = img_gray.shape
        # Focus on the center 60% of the screen for horizontal uniformity
        c1, c2 = int(w * 0.2), int(w * 0.8)
        
        # Calculate Row Stats
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        # HUC: Variance along the row
        huc_variances = np.var(img_gray[:, c1:c2], axis=1)
        
        # Binary mask: Row must be DARK and UNIFORM
        mask = np.zeros(h, dtype=np.uint8)
        for y in range(SCAN_Y_START, SCAN_Y_END):
            if intensities[y] < INTENSITY_THRESH and huc_variances[y] < HUC_VAR_THRESH:
                mask[y] = 1
        
        # Bridge text gaps (Closing)
        kernel = np.ones((40, 1), np.uint8)
        closed = cv2.morphologyEx(mask.reshape(-1, 1), cv2.MORPH_CLOSE, kernel).flatten()
        
        # Extract contiguous blocks
        candidates = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, num_labels):
            y_top = stats[i, cv2.CC_STAT_TOP]
            h_comp = stats[i, cv2.CC_STAT_HEIGHT]
            if 30 <= h_comp <= 70:
                candidates.append({'y_top': y_top, 'y_bot': y_top + h_comp})
        return candidates, intensities, huc_variances

    def update(self, img_gray):
        candidates, intensities, huc_vars = self.get_huc_candidates(img_gray)
        
        # 1. Update existing trackers with velocity-damped matching
        for trk in self.trackers:
            target_y = trk.y_top - trk.v
            best_match = None
            min_err = 35 
            
            for i, cand in enumerate(candidates):
                err = abs(cand['y_top'] - target_y)
                if err < min_err:
                    min_err = err
                    best_match = i
            
            if best_match is not None:
                match = candidates.pop(best_match)
                measured_v = trk.y_top - match['y_top']
                # Damp the velocity to keep sync
                trk.v = (trk.v * VELOCITY_SMOOTHING) + (measured_v * (1.0 - VELOCITY_SMOOTHING))
                trk.y_top = float(match['y_top'])
                trk.lost = 0
            else:
                trk.y_top -= trk.v
                trk.lost += 1
            
            if (trk.y_top + BANNER_H_TARGET) < SCAN_Y_START or trk.lost > 12:
                trk.active = False

        # 2. Spawn new trackers (only in Nucleation zone Y > 250)
        for cand in candidates:
            if cand['y_top'] > 250:
                # Prevent duplicate trackers on the same object
                if not any(abs(t.y_top - cand['y_top']) < 50 for t in self.trackers):
                    self.trackers.append(MultiTracker(cand['y_top']))

        self.trackers = [t for t in self.trackers if t.active]
        return self.trackers, intensities, huc_vars

def run_lambda_v2_lab():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    sl = SentinelLambdaV2()
    manifest = []
    
    # Process the Training Lab range (Frame 2014 center)
    for i in range(TARGET_IDX - 30, TARGET_IDX + 60):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        active_tracks, ints, huc_vars = sl.update(img_gray)
        
        overlay = img_bgr.copy()
        for trk in active_tracks:
            color = (0, 255, 0) if trk.lost == 0 else (0, 255, 255)
            cv2.rectangle(overlay, (0, int(trk.y_top)), (img_bgr.shape[1], int(trk.y_top + BANNER_H_TARGET)), color, -1)
            manifest.append({"frame": i, "id": trk.id, "y_top": trk.y_top, "v": trk.v})

        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Draw search window
        cv2.line(img_bgr, (0, SCAN_Y_START), (img_bgr.shape[1], SCAN_Y_START), (255, 0, 0), 1)
        cv2.line(img_bgr, (0, SCAN_Y_END), (img_bgr.shape[1], SCAN_Y_END), (255, 0, 0), 1)
        
        # Telemetry: Intensity (Blue) and HUC Variance (Purple)
        gx = img_bgr.shape[1] - 120
        cv2.rectangle(img_bgr, (gx, 0), (img_bgr.shape[1], img_bgr.shape[0]), (20, 20, 20), -1)
        for y in range(1, img_bgr.shape[0]):
            ix = int(gx + (ints[y]/100)*60)
            vx = int(gx + (np.log1p(huc_vars[y])/10)*60)
            cv2.line(img_bgr, (ix, y), (ix+2, y), (255, 100, 0), 1) # Intensity
            cv2.line(img_bgr, (vx, y), (vx+2, y), (255, 0, 255), 1) # HUC Var
        
        cv2.putText(img_bgr, f"LAMBDA_V2 F:{i} | TRACKS: {len(active_tracks)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"lambda_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("lambda_v2_training_manifest.csv", index=False)
    print("Lambda V2 Lab complete. Check 'lambda_v2_training_manifest.csv'.")

if __name__ == "__main__":
    run_lambda_v2_lab()