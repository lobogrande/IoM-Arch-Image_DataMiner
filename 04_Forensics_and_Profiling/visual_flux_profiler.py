import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import csv

# --- DIAGNOSTIC CONSTANTS ---
DATASET = "0"
HEADER_ROI = (52, 76, 100, 142)
GRID_ROI = (250, 550, 50, 450)

def run_full_audit():
    buffer_path = f"capture_buffer_{DATASET}"
    if not os.path.exists(buffer_path): 
        print(f"Error: {buffer_path} not found.")
        return
    
    frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
    print(f"--- AUDITING FULL RUN {DATASET} ({len(frames)} frames) ---")
    
    report = []
    prev_header = None
    prev_grid = None

    for i in range(len(frames)):
        img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
        if img is None: continue
            
        header = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        grid = cv2.resize(img[GRID_ROI[0]:GRID_ROI[1], GRID_ROI[2]:GRID_ROI[3]], (50, 50))
        
        # 1. Header Change (Movement in the Stage number area)
        h_flux = np.mean(cv2.absdiff(header, prev_header)) if prev_header is not None else 0
        
        # 2. Grid Change (Movement in the ore board)
        g_flux = np.mean(cv2.absdiff(grid, prev_grid)) if prev_grid is not None else 0
        
        # 3. Brightness (Helpful for detecting 'white-out' transitions)
        avg_brightness = np.mean(header)

        report.append({
            'frame': frames[i],
            'idx': i,
            'header_flux': round(h_flux, 3),
            'grid_flux': round(g_flux, 3),
            'brightness': round(avg_brightness, 2)
        })
        
        prev_header = header.copy()
        prev_grid = grid.copy()
        
        if i % 100 == 0: print(f" Processed {i}/{len(frames)}...")

    with open(f"full_run_audit_{DATASET}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'idx', 'header_flux', 'grid_flux', 'brightness'])
        writer.writeheader()
        writer.writerows(report)
    
    print(f"Audit Complete: full_run_audit_{DATASET}.csv generated.")

if __name__ == "__main__":
    run_full_audit()