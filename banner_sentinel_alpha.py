import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_zeta_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- CALIBRATED BOUNDS ---
SCAN_Y_START = 40   
SCAN_Y_END = 450     

# --- THRESHOLDS ---
VALLEY_THRESHOLD = 24.0  # Slightly more inclusive
GAP_FILL = 55            # Brute-force bridging for dense text
MIN_BANNER_H = 20        # Catching the very start of nucleation
MAX_BANNER_H = 90
VELOCITY_THRESHOLD = 2   # Min pixels/frame upward movement

class SentinelZeta:
    def __init__(self):
        self.last_y = None
        self.ui_blacklist = set() # Y-coordinates identified as stationary
        self.history = [] # List of confirmed banner Y positions

    def detect_and_validate(self, img_gray, frame_idx):
        h, w = img_gray.shape
        c1, c2 = int(w * 0.35), int(w * 0.65)
        intensities = np.mean(img_gray[:, c1:c2], axis=1)
        
        # 1. Generate Raw Candidates
        mask = np.zeros(h, dtype=np.uint8)
        mask[SCAN_Y_START:SCAN_Y_END] = (intensities[SCAN_Y_START:SCAN_Y_END] < VALLEY_THRESHOLD)
        
        kernel = np.ones((GAP_FILL, 1), np.uint8)
        closed_mask = cv2.morphologyEx(mask.reshape(-1, 1), cv2.MORPH_CLOSE, kernel).flatten()
        
        candidates = []
        start_y = None
        for y, val in enumerate(closed_mask):
            if val == 1 and start_y is None: start_y = y
            elif val == 0 and start_y is not None:
                height = y - start_y
                if MIN_BANNER_H <= height <= MAX_BANNER_H:
                    candidates.append({'top': start_y, 'bot': y, 'center': (start_y + y) // 2})
                start_y = None

        # 2. MOTION VALIDATION
        confirmed_banners = []
        for cand in candidates:
            # Check if this Y is a known stationary HUD area
            if any(abs(cand['center'] - b_y) < 5 for b_y in self.ui_blacklist):
                continue

            if self.last_y is not None:
                velocity = self.last_y - cand['center']
                # If it moved up, it's a banner. 
                # If it's stationary (0 movement) for several frames, blacklist it.
                if velocity >= VELOCITY_THRESHOLD:
                    cand['state'] = "NUC" if np.mean(img_gray[cand['top']:cand['bot'], :]) > 15 else "FULL"
                    confirmed_banners.append(cand)
                    self.last_y = cand['center']
                elif abs(velocity) < 2:
                    # Potential UI element. We don't blacklist immediately to allow for slow starts.
                    pass 
            else:
                # First frame initialization
                self.last_y = cand['center']
                cand['state'] = "NUC"
                confirmed_banners.append(cand)

        # 3. BLACKLIST LOGIC: If a candidate persists at one spot, kill it.
        # (For this experiment, we'll manually block the known 160-180 UI area)
        if any(abs(cand['center'] - 180) < 15 for cand in candidates):
            self.ui_blacklist.add(180)

        return confirmed_banners, intensities

def run_sentinel_zeta():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170

    sz = SentinelZeta()
    manifest = []
    
    for i in range(target_idx - 15, target_idx + 60):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        zones, intensities = sz.detect_and_validate(img_gray, i)
        
        # Visuals
        overlay = img_bgr.copy()
        for z in zones:
            color = (0, 255, 255) if z['state'] == "NUC" else (0, 0, 255)
            cv2.rectangle(overlay, (0, z['top']), (img_bgr.shape[1], z['bot']), color, -1)
            manifest.append({"idx": i, "y_top": z['top'], "y_bot": z['bot'], "state": z['state']})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # UI Blacklist Visualization (Grey lines)
        for b_y in sz.ui_blacklist:
            cv2.line(img_bgr, (0, b_y), (100, b_y), (100, 100, 100), 2)

        # Draw Telemetry and Ruler
        # (Assuming helper functions draw_pixel_ruler and draw_telemetry are available)
        cv2.putText(img_bgr, f"ZETA F:{i} | BANNERS: {len(zones)} | UI_LOCK: {len(sz.ui_blacklist)}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"zeta_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_zeta_manifest.csv", index=False)
    print(f"Sentinel Zeta complete. Temporal motion validation active.")

if __name__ == "__main__":
    run_sentinel_zeta()