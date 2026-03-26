import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import csv

# --- MASTER CONSTANTS ---
DATASET = "0"
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)

# Problem Zones to Profile
ZONES = [
    (2550, 4000),   # F34-42 (The F39 Skip)
    (7450, 9500),   # F63-71 (The F67 Skip)
    (17200, 18500)  # F101-105 (The F103 Dupe)
]

def run_hyper_scanner():
    buffer_path = f"capture_buffer_{DATASET}"
    frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
    
    analysis_log = []
    print(f"--- STARTING HYPER-SCAN ON DATASET {DATASET} ---")

    for start, end in ZONES:
        print(f" Scanning Frames {start} to {end}...")
        
        # Initial state for this zone
        prev_img = cv2.imread(os.path.join(buffer_path, frames[start]), 0)
        prev_h = prev_img[52:76, 100:142]
        prev_c = prev_img[230:246, 250:281]
        
        # The 'Anchor' is the state of the starting floor
        anchor_h = prev_h.copy()
        
        for i in range(start + 1, min(end, len(frames))):
            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            if img is None: continue
            
            curr_h = img[52:76, 100:142]
            curr_c = img[230:246, 250:281]
            
            # 1. FLUX (The Pulse)
            h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
            c_flux = np.mean(cv2.absdiff(curr_c, prev_c))
            
            # 2. IDENTITY (Similarity vs Anchor)
            res = cv2.matchTemplate(curr_h, anchor_h, cv2.TM_CCORR_NORMED)
            identity = res.max()
            
            # 3. DELTA (Literal Pixel Change vs Anchor)
            anc_mae = np.mean(cv2.absdiff(curr_h, anchor_h))
            
            # 4. STRUCTURAL SCORE (Vertical Projection Variance)
            row_sums = np.sum(cv2.absdiff(curr_h, prev_h), axis=1)
            struct_v = np.std(row_sums)
            
            # 5. BRIGHTNESS
            brightness = np.mean(img)

            analysis_log.append({
                'idx': i,
                'h_flux': round(h_flux, 4),
                'c_flux': round(c_flux, 4),
                'identity': round(identity, 4),
                'anc_mae': round(anc_mae, 4),
                'struct_v': round(struct_v, 2),
                'brightness': round(brightness, 2)
            })
            
            # Note: We do NOT update the anchor here, because we want to see 
            # how the metrics evolve relative to the 'Start' of the floor.
            prev_h, prev_c = curr_h, curr_c

    with open(f"hyper_signal_analysis_{DATASET}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'h_flux', 'c_flux', 'identity', 'anc_mae', 'struct_v', 'brightness'])
        writer.writeheader()
        writer.writerows(analysis_log)
    
    print(f"Scan complete. hyper_signal_analysis_{DATASET}.csv generated.")

if __name__ == "__main__":
    run_hyper_scanner()