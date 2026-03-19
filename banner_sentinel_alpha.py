import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- CALIBRATED THRESHOLDS (From your Forensic CSV) ---
VAR_THRESH = 450      # Anything flatter than this is "suspicious"
INT_THRESH = 12       # Anything darker than this is "black rectangle"
MIN_BANNER_H = 30     # Minimum vertical height to be a banner
MAX_BANNER_H = 65     # Maximum vertical height

def detect_banner_zones(img_gray):
    """Identifies Y-ranges occupied by banners based on HPP and Variance."""
    hpp = np.mean(img_gray, axis=1)
    var = np.var(img_gray, axis=1)
    
    # Identify "Candidate Rows"
    # Logic: Row must be dark AND flat
    candidate_mask = (hpp < INT_THRESH) & (var < VAR_THRESH)
    
    # Cluster contiguous rows into blocks
    zones = []
    start_y = None
    
    for y, is_candidate in enumerate(candidate_mask):
        if is_candidate and start_y is None:
            start_y = y
        elif not is_candidate and start_y is not None:
            height = y - start_y
            if MIN_BANNER_H <= height <= MAX_BANNER_H:
                zones.append((start_y, y))
            start_y = None
            
    return zones

def run_sentinel_alpha():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try:
        target_idx = all_files.index(TARGET_FILENAME)
    except ValueError:
        target_idx = 1170 # Fallback
        
    # Process 60 frames around the event
    manifest = []
    for i in range(target_idx - 5, target_idx + 55):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        zones = detect_banner_zones(img_gray)
        
        # Visual Audit: Draw the mask
        overlay = img_bgr.copy()
        for (y_top, y_bot) in zones:
            cv2.rectangle(overlay, (0, y_top), (img_bgr.shape[1], y_bot), (0, 0, 255), -1)
            manifest.append({"idx": i, "filename": fname, "y_top": y_top, "y_bot": y_bot})
        
        # Blend the overlay (50% transparency)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, img_bgr)
        
        # Label the frame
        cv2.putText(img_bgr, f"Frame: {i} | Banners: {len(zones)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"debug_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_alpha_results.csv", index=False)
    print(f"Sentinel Alpha finished. Check the '{OUT_DIR}' folder for visuals.")

if __name__ == "__main__":
    run_sentinel_alpha()