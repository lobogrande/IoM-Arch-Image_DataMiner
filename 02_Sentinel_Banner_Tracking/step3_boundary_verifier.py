# step3_boundary_verifier.py
# Purpose: Master Plan Step 3 - Finalize floor boundaries by scanning backward 
#          from Step 2 anchors to find the exact DNA shift frame.
# Version: 1.1 (The Boundary Scalpel - KeyError Fix)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
CANDIDATES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "boundary_verification")

# DNA SENSOR CONSTANTS (Matching dna_sensor_audit.py)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
VALLEY_THRESHOLD = 0.75

def load_bg_templates():
    templates = []
    # Background and UI elements that represent "Empty" slots
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"background_plain_{i}.png")
        if os.path.exists(p): templates.append(cv2.imread(p, 0))
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"negative_ui_{i}.png")
        if os.path.exists(p): templates.append(cv2.imread(p, 0))
    return [t for t in templates if t is not None]

def get_frame_dna(img_gray, templates):
    """Detects 12-bit DNA for a single frame."""
    def get_bit(r_idx, c_idx):
        y = int(ORE0_Y + (r_idx * STEP))
        x = int(ORE0_X + (c_idx * STEP))
        tw, th = 30, 30
        roi = img_gray[y-15:y+15, x-15:x+15]
        if roi.shape != (30, 30): return '1'
        
        best_val = -1
        for t in templates:
            res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            best_val = max(best_val, cv2.minMaxLoc(res)[1])
        return '0' if best_val >= VALLEY_THRESHOLD else '1'

    r3 = "".join([get_bit(2, c) for c in range(6)])
    r4 = "".join([get_bit(3, c) for c in range(6)])
    return f"{r4}-{r3}"

def run_boundary_verification():
    if not os.path.exists(CANDIDATES_CSV):
        print(f"Error: {CANDIDATES_CSV} not found. Run Step 2 chunker first.")
        return

    df = pd.read_csv(CANDIDATES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    bg_tpls = load_bg_templates()
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 3: BOUNDARY VERIFICATION (110 Floors) ---")
    
    final_boundaries = []
    
    for i, row in df.iterrows():
        floor_id = int(row['floor_id'])
        anchor_idx = int(row['start_frame'])
        target_dna = f"{row['r4_dna_stable']}-{row['r3_dna_stable']}"
        
        # Determine how far back we can scan
        limit_idx = 0
        if i > 0:
            # Fix: Use 'true_start_frame' instead of 'start_frame'
            limit_idx = final_boundaries[-1]['true_start_frame'] + 1
            
        print(f"Floor {floor_id:03d}: Scanning back from {anchor_idx} (Limit: {limit_idx})...", end="\r")
        
        true_start = anchor_idx
        # Scan backward
        for b_idx in range(anchor_idx - 1, limit_idx - 1, -1):
            img = cv2.imread(os.path.join(buffer_dir, all_files[b_idx]), 0)
            if img is None: break
            
            current_dna = get_frame_dna(img, bg_tpls)
            if current_dna == target_dna:
                true_start = b_idx # Still on the same floor
            else:
                break # Found the DNA shift!
                
        # Record boundary
        floor_data = {
            'floor_id': floor_id,
            'true_start_frame': true_start,
            'anchor_frame': anchor_idx,
            'dna_sig': target_dna
        }
        
        # Link to previous floor's end
        if i > 0:
            final_boundaries[-1]['end_frame'] = true_start - 1
            final_boundaries[-1]['duration'] = final_boundaries[-1]['end_frame'] - final_boundaries[-1]['true_start_frame'] + 1
            
        final_boundaries.append(floor_data)

    # Handle the very last floor (end of buffer)
    final_boundaries[-1]['end_frame'] = len(all_files) - 1
    final_boundaries[-1]['duration'] = final_boundaries[-1]['end_frame'] - final_boundaries[-1]['true_start_frame'] + 1

    # Save and Report
    out_df = pd.DataFrame(final_boundaries)
    out_df.to_csv(OUT_CSV, index=False)
    
    print(f"\n[DONE] Verified {len(out_df)} floor boundaries.")
    print(f"Saved results to: {OUT_CSV}")

    # Visual proof of a few boundaries
    for i in [0, 30, 72, 109]:
        if i >= len(final_boundaries): continue
        f = final_boundaries[i]
        img = cv2.imread(os.path.join(buffer_dir, all_files[f['true_start_frame']]))
        if img is not None:
            cv2.rectangle(img, (10, img.shape[0]-60), (500, img.shape[0]-10), (0,0,0), -1)
            cv2.putText(img, f"FLOOR {f['floor_id']} TRUE START | Frame: {f['true_start_frame']}", (20, img.shape[0]-35), 0, 0.6, (0,255,0), 2)
            cv2.imwrite(os.path.join(VERIFY_DIR, f"boundary_floor_{f['floor_id']:03d}_start.jpg"), img)

if __name__ == "__main__":
    run_boundary_verification()