# diag_boundary_integrity.py
# Purpose: Programmatically detect "Missed Starts" by finding grid presence 
#          in unassigned timeline gaps.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")

def get_frame_complexity(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def run_integrity_diagnostic():
    if not os.path.exists(BOUNDARIES_CSV):
        print("Error: Run step3_boundary_verifier.py first.")
        return

    df = pd.read_csv(BOUNDARIES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- BOUNDARY INTEGRITY DIAGNOSTIC ---")
    print("Searching for 'Orphaned Grid' frames between floor boundaries...\n")

    orphans = []
    
    # Analyze the gaps between each floor's Start and the previous floor's End
    for i in range(1, len(df)):
        prev_end = int(df.iloc[i-1]['anchor_frame']) # Use anchor to find mining gap
        curr_start = int(df.iloc[i]['true_start_frame'])
        
        if curr_start - prev_end > 5:
            # We have a gap where we aren't mining. Check if the grid is there.
            for idx in range(prev_end + 1, curr_start):
                img = cv2.imread(os.path.join(buffer_dir, all_files[idx]), 0)
                # If complexity is high in the grid area, the floor was already present
                comp = get_frame_complexity(img[255:490, :])
                if comp > 500:
                    orphans.append({'frame': idx, 'floor_after': df.iloc[i]['floor_id'], 'complexity': comp})

    if orphans:
        orphan_df = pd.DataFrame(orphans)
        print(f"WARNING: Detected {len(orphans)} frames with grid presence that were missed.")
        print(f"Heaviest Missed Starts detected near floors: {orphan_df['floor_after'].unique()}")
        orphan_df.to_csv("orphaned_grid_frames.csv", index=False)
    else:
        print("PASS: No significant orphaned grid frames detected.")

if __name__ == "__main__":
    run_integrity_diagnostic()