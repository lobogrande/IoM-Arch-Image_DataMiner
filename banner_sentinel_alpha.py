import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_regional_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- CALIBRATED REGIONAL THRESHOLDS ---
# Nucleation (Center)
C_INT_THRESH = 1.5    
C_VAR_THRESH = 1.0    
# Full Width (Expansion)
F_INT_THRESH = 10.0   
F_VAR_THRESH = 400.0  

MIN_H, MAX_H = 25, 70

def detect_regional_zones(img_gray):
    h, w = img_gray.shape
    c1, c2 = int(w * 0.35), int(w * 0.65)
    
    # Split Regions
    center_roi = img_gray[:, c1:c2]
    
    # Row-wise stats for Center
    c_hpp = np.mean(center_roi, axis=1)
    c_var = np.var(center_roi, axis=1)
    
    # Row-wise stats for Full Row
    f_hpp = np.mean(img_gray, axis=1)
    f_var = np.var(img_gray, axis=1)
    
    zones = []
    start_y = None
    state = None
    
    for y in range(h):
        # State 1: Nucleation (Center is black/flat, but full row might not be)
        is_nuc = (c_hpp[y] < C_INT_THRESH) and (c_var[y] < C_VAR_THRESH)
        # State 2: Full Width (Entire row is relatively black/flat)
        is_full = (f_hpp[y] < F_INT_THRESH) and (f_var[y] < F_VAR_THRESH)
        
        if (is_nuc or is_full) and start_y is None:
            start_y = y
            state = "FULL" if is_full else "NUC"
        elif not (is_nuc or is_full) and start_y is not None:
            height = y - start_y
            if MIN_H <= height <= MAX_H:
                zones.append({'top': start_y, 'bot': y, 'state': state})
            start_y = None
            
    return zones

def run_regional_sentinel():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170
        
    manifest = []
    # Process from the first nucleation frame (approx 15 frames before the target)
    for i in range(target_idx - 15, target_idx + 45):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        zones = detect_regional_zones(img_gray)
        
        overlay = img_bgr.copy()
        for z in zones:
            color = (0, 255, 255) if z['state'] == "NUC" else (0, 0, 255) # Yellow vs Red
            cv2.rectangle(overlay, (0, z['top']), (w, z['bot']), color, -1)
            manifest.append({"idx": i, "y_top": z['top'], "y_bot": z['bot'], "state": z['state']})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        cv2.putText(img_bgr, f"Frame: {i} | Banners: {len(zones)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"regional_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_regional_results.csv", index=False)
    print(f"Regional Sentinel finished. Check '{OUT_DIR}' for Nucleation(Yellow) and Full(Red) masks.")

if __name__ == "__main__":
    run_regional_sentinel()