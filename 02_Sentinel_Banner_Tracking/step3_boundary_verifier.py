# step3_boundary_verifier.py
# Purpose: Master Plan Step 3 - Finalize floor boundaries by scanning backward 
#          from Step 2 anchors to find the exact DNA shift frame.
# Version: 1.5 (The Context-Aware Scalpel: Preventing Boss Leaks)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
CANDIDATES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_start_candidates.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
REPORT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "boundary_integrity_report.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "boundary_verification")

# TARGET TRUTH SET (For console highlighting and visual export)
TRUTH_SET = [70, 71, 77, 78, 96, 98, 99]

# DNA SENSOR CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
VALLEY_THRESHOLD = 0.75
TUNNEL_LIMIT = 10 

def load_bg_templates():
    templates = []
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"background_plain_{i}.png")
        if os.path.exists(p): templates.append(cv2.imread(p, 0))
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"negative_ui_{i}.png")
        if os.path.exists(p): templates.append(cv2.imread(p, 0))
    return [t for t in templates if t is not None]

def get_frame_dna(img_gray, templates):
    """Detects 12-bit DNA for a single frame, forcing 6-bit padding."""
    def get_bit(r_idx, c_idx):
        y = int(ORE0_Y + (r_idx * STEP))
        x = int(ORE0_X + (c_idx * STEP))
        roi = img_gray[y-15:y+15, x-15:x+15]
        if roi.shape != (30, 30): return '1'
        best_val = -1
        for t in templates:
            res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            best_val = max(best_val, cv2.minMaxLoc(res)[1])
        return '0' if best_val >= VALLEY_THRESHOLD else '1'

    r3 = "".join([get_bit(2, c) for c in range(6)]).zfill(6)
    r4 = "".join([get_bit(3, c) for c in range(6)]).zfill(6)
    return f"{r4}-{r3}"

def run_boundary_verification():
    if not os.path.exists(CANDIDATES_CSV):
        print(f"Error: {CANDIDATES_CSV} not found.")
        return

    df = pd.read_csv(CANDIDATES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    bg_tpls = load_bg_templates()
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 3: BOUNDARY VERIFICATION v1.5 ---")
    
    final_boundaries = []
    integrity_logs = []
    
    for i, row in df.iterrows():
        floor_id = int(row['floor_id'])
        anchor_idx = int(row['start_frame'])
        
        target_r4 = str(row['r4_dna_stable']).zfill(6)
        target_r3 = str(row['r3_dna_stable']).zfill(6)
        target_dna = f"{target_r4}-{target_r3}"
        
        # Determine scanning limit
        limit_idx = 0
        is_collision = False
        if i > 0:
            prev_r4 = str(df.iloc[i-1]['r4_dna_stable']).zfill(6)
            prev_r3 = str(df.iloc[i-1]['r3_dna_stable']).zfill(6)
            prev_dna = f"{prev_r4}-{prev_r3}"
            
            # BOSS COLLISION CIRCUIT BREAKER
            # If DNA matches, we cannot scan back into the previous floor's anchor zone.
            if target_dna == prev_dna:
                is_collision = True
                # Allow only a 15-frame backtrack to find the grid appearance
                limit_idx = max(int(df.iloc[i-1]['start_frame']) + 20, anchor_idx - 15)
            else:
                limit_idx = int(df.iloc[i-1]['start_frame']) + 1
            
        true_start = anchor_idx
        b_idx = anchor_idx - 1
        
        while b_idx >= limit_idx:
            if b_idx % 50 == 0 or floor_id in TRUTH_SET:
                tag = " [BOSS COLLISION]" if is_collision else ""
                print(f"Floor {floor_id:03d}{tag}: Scanning back {anchor_idx} -> {b_idx}...", end="\r")
            
            img = cv2.imread(os.path.join(buffer_dir, all_files[b_idx]), 0)
            if img is None: break
            
            dna = get_frame_dna(img, bg_tpls)
            
            if dna == target_dna:
                true_start = b_idx
                b_idx -= 1
            else:
                found_reversion = False
                for look_idx in range(b_idx - 1, max(limit_idx - 1, b_idx - TUNNEL_LIMIT), -1):
                    chk_img = cv2.imread(os.path.join(buffer_dir, all_files[look_idx]), 0)
                    if chk_img is None: break
                    if get_frame_dna(chk_img, bg_tpls) == target_dna:
                        found_reversion = True
                        true_start = look_idx
                        b_idx = look_idx - 1
                        break
                if not found_reversion: break
        
        floor_data = {
            'floor_id': floor_id,
            'true_start_frame': true_start,
            'anchor_frame': anchor_idx,
            'dna_sig': target_dna
        }
        
        integrity_logs.append({
            'floor_id': floor_id,
            'shift_dist': anchor_idx - true_start,
            'dna_sig': target_dna
        })
        
        if i > 0:
            final_boundaries[-1]['end_frame'] = true_start - 1
            final_boundaries[-1]['duration'] = final_boundaries[-1]['end_frame'] - final_boundaries[-1]['true_start_frame'] + 1
            
        final_boundaries.append(floor_data)

    final_boundaries[-1]['end_frame'] = len(all_files) - 1
    final_boundaries[-1]['duration'] = final_boundaries[-1]['end_frame'] - final_boundaries[-1]['true_start_frame'] + 1

    pd.DataFrame(final_boundaries).to_csv(OUT_CSV, index=False)
    pd.DataFrame(integrity_logs).to_csv(REPORT_CSV, index=False)
    
    print(f"\n[DONE] Verified {len(final_boundaries)} floors. Results saved to {OUT_CSV}")
    
    # VISUAL AUDIT: Generate images for critical floors
    print("\nGenerating visual verification proofs...")
    log_df = pd.DataFrame(integrity_logs)
    for idx, f in enumerate(final_boundaries):
        # We output images for our manual truth set AND any floor that actually shifted significantly
        dist = int(log_df[log_df['floor_id'] == f['floor_id']].iloc[0]['shift_dist'])
        if f['floor_id'] in TRUTH_SET or dist > 5:
            img = cv2.imread(os.path.join(buffer_dir, all_files[f['true_start_frame']]))
            if img is not None:
                h, w = img.shape[:2]
                cv2.rectangle(img, (10, h-65), (700, h-10), (0,0,0), -1)
                cv2.putText(img, f"FLOOR {f['floor_id']:03d} TRUE START | Frame: {f['true_start_frame']}", (20, h-40), 0, 0.6, (0,255,0), 2)
                cv2.putText(img, f"Shifted: {dist} frames from anchor", (20, h-15), 0, 0.5, (255,255,255), 1)
                cv2.imwrite(os.path.join(VERIFY_DIR, f"boundary_f{f['floor_id']:03d}_start.jpg"), img)

if __name__ == "__main__":
    run_boundary_verification()