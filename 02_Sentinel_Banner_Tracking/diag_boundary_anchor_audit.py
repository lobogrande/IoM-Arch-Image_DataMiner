# diag_boundary_anchor_audit.py
# Purpose: Generate side-by-side comparisons of Candidate Start Frames (N) 
#          and their predecessors (N-1) to identify missed floor starts.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
CANDIDATES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_verification", "transition_audit")

def run_anchor_audit():
    if not os.path.exists(CANDIDATES_CSV):
        print(f"Error: {CANDIDATES_CSV} not found.")
        return

    # Load floor candidates
    df = pd.read_csv(CANDIDATES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    
    # Get all files to find N-1 easily
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    print(f"--- FLOOR ANCHOR TRANSITION AUDIT ---")
    print(f"Generating side-by-side comparisons for {len(df)} floors...")

    for i, row in df.iterrows():
        floor_id = int(row['floor_id'])
        curr_idx = int(row['start_frame'])
        
        # We can't look back from floor 1 (frame 0)
        if curr_idx == 0:
            print(f"  Floor {floor_id:03d}: Skipped (Start Frame is 0)")
            continue
            
        prev_idx = curr_idx - 1
        
        # Load images
        img_curr = cv2.imread(os.path.join(buffer_dir, all_files[curr_idx]))
        img_prev = cv2.imread(os.path.join(buffer_dir, all_files[prev_idx]))
        
        if img_curr is None or img_prev is None:
            print(f"  Floor {floor_id:03d}: Error loading frames {prev_idx}/{curr_idx}")
            continue
            
        # Ensure they are the same height for hstack
        h1, w1 = img_prev.shape[:2]
        h2, w2 = img_curr.shape[:2]
        if h1 != h2:
            img_curr = cv2.resize(img_curr, (int(w2 * h1/h2), h1))
            
        # Annotate frames
        cv2.putText(img_prev, f"PREV (F{prev_idx})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(img_curr, f"START (F{curr_idx})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Combine side-by-side
        combined = np.hstack((img_prev, img_curr))
        
        # Downscale for easier viewing (optional)
        display_w = 1600
        aspect = combined.shape[0] / combined.shape[1]
        combined_res = cv2.resize(combined, (display_w, int(display_w * aspect)))
        
        # Add Floor ID header to the combined image
        cv2.rectangle(combined_res, (0, 0), (350, 60), (0, 0, 0), -1)
        cv2.putText(combined_res, f"FLOOR {floor_id:03d}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        # Save
        out_name = f"floor_{floor_id:03d}_transition_N{curr_idx}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, out_name), combined_res)
        
        if floor_id % 10 == 0:
            print(f"  Processed {floor_id}/110 floors...")

    print(f"\n[DONE] Audit images saved to: {OUT_DIR}")

if __name__ == "__main__":
    run_anchor_audit()