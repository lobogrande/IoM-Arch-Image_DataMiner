import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_beta_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- CALIBRATED THRESHOLDS ---
# Based on the Valley Profile, we allow rows up to 20.0 intensity to be 
# part of the "valley" to ensure we catch the edges correctly.
VALLEY_THRESHOLD = 20.0  
GAP_BRIDGE_SIZE = 45     # Max vertical pixels to bridge (covers text/icons)
MIN_BANNER_H = 35        # Minimum height of the bridged object
MAX_BANNER_H = 75        # Maximum height of the bridged object

def detect_bridged_banners(img_gray):
    h, w = img_gray.shape
    # We focus detection on the Center ROI (35%-65%) to catch nucleation early
    c1, c2 = int(w * 0.35), int(w * 0.65)
    center_roi = img_gray[:, c1:c2]
    
    # 1. Calculate the HPP (Mean Intensity) of the center strip
    intensities = np.mean(center_roi, axis=1)
    
    # 2. Create a Binary Mask of "Dark" rows
    # 1 = Potential Banner Row, 0 = Background/Noise
    mask = (intensities < VALLEY_THRESHOLD).astype(np.uint8)
    
    # 3. Apply Morphological Closing
    # This 'smears' the detection vertically, filling in any gaps (like white text)
    # that are smaller than GAP_BRIDGE_SIZE.
    kernel = np.ones((GAP_BRIDGE_SIZE, 1), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    zones = []
    start_y = None
    
    # 4. Extract contiguous blocks from the closed mask
    for y, is_active in enumerate(closed_mask):
        if is_active == 1 and start_y is None:
            start_y = y
        elif is_active == 0 and start_y is not None:
            height = y - start_y
            if MIN_BANNER_H <= height <= MAX_BANNER_H:
                # Double-check the "Nucleation" state:
                # If the full row is still bright, it's an incipient (NUC) banner
                full_row_avg = np.mean(img_gray[start_y:y, :])
                state = "NUC" if full_row_avg > 15.0 else "FULL"
                zones.append({'top': start_y, 'bot': y, 'state': state})
            start_y = None
            
    return zones

def run_sentinel_beta():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try:
        target_idx = all_files.index(TARGET_FILENAME)
        print(f"Target found at Index: {target_idx}")
    except ValueError:
        target_idx = 1170
        print(f"Target not found, starting at default index {target_idx}")

    manifest = []
    # Analyze 15 frames before and 50 frames after to see the full lifecycle
    for i in range(target_idx - 15, target_idx + 50):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        h, w, _ = img_bgr.shape
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        zones = detect_bridged_banners(img_gray)
        
        # Draw Visuals
        overlay = img_bgr.copy()
        for z in zones:
            # Yellow = Nucleation (Center-only), Red = Mature (Full-width)
            color = (0, 255, 255) if z['state'] == "NUC" else (0, 0, 255)
            cv2.rectangle(overlay, (0, z['top']), (w, z['bot']), color, -1)
            manifest.append({
                "idx": i, 
                "filename": fname, 
                "y_top": z['top'], 
                "y_bot": z['bot'], 
                "state": z['state']
            })
            
        # Alpha blend for translucency
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Metadata Overlay
        cv2.putText(img_bgr, f"FRAME: {i} | BANNERS: {len(zones)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"beta_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_beta_manifest.csv", index=False)
    print(f"Sentinel Beta complete. Images saved to '{OUT_DIR}'.")

if __name__ == "__main__":
    run_sentinel_beta()