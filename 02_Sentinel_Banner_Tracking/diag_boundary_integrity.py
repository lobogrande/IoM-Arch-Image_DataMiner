# diag_boundary_integrity.py
# Purpose: Programmatically detect "Missed Starts" by finding grid presence 
#          in unassigned timeline gaps.
# Version: 1.3 (Progress Reporting & Boundary Alignment)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")

def get_frame_complexity(img_gray):
    """Calculates Laplacian variance to detect the presence of sharp grid lines/ores."""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def run_integrity_diagnostic():
    if not os.path.exists(BOUNDARIES_CSV):
        print("Error: Run step3_boundary_verifier.py first.")
        return

    df = pd.read_csv(BOUNDARIES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- BOUNDARY INTEGRITY DIAGNOSTIC v1.3 ---")
    print(f"Analyzing gaps between {len(df)} floors for orphaned grid frames...")

    orphans = []
    total_gap_frames = 0
    
    # 1. Analyze the timeline gaps
    # A "Gap" is the space between Floor N's end and Floor N+1's start.
    for i in range(len(df)):
        # Determine the start and end of the search window
        if i == 0:
            start_search = 0
            end_search = int(df.iloc[i]['true_start_frame'])
        else:
            start_search = int(df.iloc[i-1]['end_frame']) + 1
            end_search = int(df.iloc[i]['true_start_frame'])
            
        gap_size = end_search - start_search
        if gap_size <= 0:
            continue
            
        total_gap_frames += gap_size
        
        # Scan the gap
        for idx in range(start_search, end_search):
            # Progress reporting
            if idx % 500 == 0:
                print(f"  Scanning Frame {idx}/{len(all_files)}...", end="\r")
                
            img = cv2.imread(os.path.join(buffer_dir, all_files[idx]), 0)
            if img is None: continue
            
            # Focus complexity check on the Row 3/4 Grid Area (Y: 255 to 490)
            roi = img[255:490, :]
            comp = get_frame_complexity(roi)
            
            # A complexity > 600 strongly indicates the grid/ores are present
            if comp > 600:
                orphans.append({
                    'frame': idx, 
                    'floor_after': df.iloc[i]['floor_id'], 
                    'complexity': round(comp, 2)
                })

    print(f"\n\n--- DIAGNOSTIC COMPLETE ---")
    if orphans:
        orphan_df = pd.DataFrame(orphans)
        print(f"WARNING: Detected {len(orphans)} orphaned grid frames (out of {total_gap_frames} checked).")
        print(f"Major missed starts likely near floors: {orphan_df['floor_after'].unique()}")
        
        out_path = os.path.join(cfg.DATA_DIRS["TRACKING"], "orphaned_grid_frames.csv")
        orphan_df.to_csv(out_path, index=False)
        print(f"Full orphan report saved to: {out_path}")
    else:
        print("PASS: No significant orphaned grid frames detected in the timeline gaps.")

if __name__ == "__main__":
    run_integrity_diagnostic()