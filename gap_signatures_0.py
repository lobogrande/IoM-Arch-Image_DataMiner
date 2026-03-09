import cv2
import numpy as np
import os
import csv

# --- DIAGNOSTIC CONSTANTS ---
DATASET = "0"
HEADER_ROI = (52, 76, 100, 142)
GRID_ROI = (250, 550, 50, 450)

def run_gradient_diagnostic():
    buffer_path = f"capture_buffer_{DATASET}"
    frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
    
    # PROBLEM AREAS TO PROFILE: 
    # Gap 1: F7 -> F10 (Indices 122 - 171)
    # Gap 2: F17 -> F20 (Indices 742 - 860)
    focus_zones = [(120, 175), (740, 870)]
    
    report = []
    print(f"--- PROFILING PROBLEM ZONES ---")

    for start, end in focus_zones:
        # Get baseline from the anchor at 'start'
        base_gray = cv2.imread(os.path.join(buffer_path, frames[start]), 0)
        base_h = base_gray[52:76, 100:142]
        
        for i in range(start, end):
            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[52:76, 100:142]
            
            # MSE vs Anchor (How far have we moved from the starting floor?)
            mse_anchor = np.mean(cv2.absdiff(curr_h, base_h))
            
            # Frame-to-Frame Flux (The 'Pulse')
            if i > start:
                prev_h = cv2.imread(os.path.join(buffer_path, frames[i-1]), 0)[52:76, 100:142]
                flux = np.mean(cv2.absdiff(curr_h, prev_h))
            else:
                flux = 0
                
            # Standard Deviation of Header (Is it 'noisy' or 'clean'?)
            header_std = np.std(curr_h)

            report.append({
                'idx': i,
                'zone': f"{start}-{end}",
                'mse_anchor': round(mse_anchor, 3),
                'flux': round(flux, 3),
                'header_std': round(header_std, 3)
            })

    with open(f"gap_signatures_{DATASET}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['idx', 'zone', 'mse_anchor', 'flux', 'header_std'])
        writer.writeheader()
        writer.writerows(report)
    
    print(f"Done. Please upload gap_signatures_{DATASET}.csv")

if __name__ == "__main__":
    run_gradient_diagnostic()