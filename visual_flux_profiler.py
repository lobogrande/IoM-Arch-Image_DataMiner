import cv2
import numpy as np
import os
import json
import csv

# MASTER CONSTANTS
DATASET = "0" 
HEADER_ROI = (52, 76, 100, 142)
GRID_BASE_ROI = (485, 500, 75, 425) # Just the bottom strip of the grid

def run_flux_profiler():
    json_file = f"milestones_run_{DATASET}.json"
    buffer_path = f"capture_buffer_{DATASET}"
    if not os.path.exists(json_file): return
    
    with open(json_file, 'r') as f: anchors = json.load(f)
    frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
    
    # We will profile the gap between F7 and F10 (Frames 122 - 157 in your log)
    start_idx, end_idx = 122, 160 
    
    # Extract baseline signature from F7 anchor
    base_img = cv2.imread(os.path.join(buffer_path, frames[start_idx]), 0)
    base_header = base_img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    
    print(f"--- PROFILING GAP F7->F10 (Frames {start_idx} to {end_idx}) ---")
    
    report = []
    prev_header = None
    prev_grid = None

    for i in range(start_idx, end_idx):
        img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
        header = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        grid = img[GRID_BASE_ROI[0]:GRID_BASE_ROI[1], GRID_BASE_ROI[2]:GRID_BASE_ROI[3]]
        
        # 1. Similarity to Anchor (MSE)
        anchor_diff = np.mean(cv2.absdiff(header, base_header))
        
        # 2. Frame-to-Frame Jitter (Flux)
        header_flux = np.mean(cv2.absdiff(header, prev_header)) if prev_header is not None else 0
        grid_flux = np.mean(cv2.absdiff(grid, prev_grid)) if prev_grid is not None else 0
        
        report.append({
            'frame': frames[i],
            'idx': i,
            'anchor_diff': round(anchor_diff, 2),
            'header_flux': round(header_flux, 2),
            'grid_flux': round(grid_flux, 2)
        })
        
        prev_header = header.copy()
        prev_grid = grid.copy()

    # Save to CSV for analysis
    with open(f"flux_report_run_{DATASET}.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'idx', 'anchor_diff', 'header_flux', 'grid_flux'])
        writer.writeheader()
        writer.writerows(report)
    
    print(f"Diagnostic Complete. Please look at flux_report_run_{DATASET}.csv")

if __name__ == "__main__":
    run_flux_profiler()