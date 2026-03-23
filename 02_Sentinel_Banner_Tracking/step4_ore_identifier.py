# step4_ore_identifier.py
# Purpose: Master Plan Step 4.1 - Establish 100% accurate 24-slot DNA Occupancy
#          using Temporal Sliding-Window Scanning and Diagnostic Auditing.
# Version: 1.6 (Production Release - All Floors)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
DIAG_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_score_analysis.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_proofs")

# DIAGNOSTIC CONTROL
LIMIT_FLOORS = None  # Process all 110 floors for production

# GRID CONSTANTS (Ore Centers)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

# THRESHOLDS
# Logic: High match with background = Empty (0). 
EMPTY_THRESHOLD = 0.75
MAX_DNA_WINDOW = 150 # Max frames to scan per floor for occupancy

def load_bg_templates():
    """Loads background and negative UI templates (48x48)."""
    templates = []
    t_path = cfg.TEMPLATE_DIR
    for i in range(10):
        p_bg = os.path.join(t_path, f"background_plain_{i}.png")
        if os.path.exists(p_bg): templates.append(cv2.imread(p_bg, 0))
        p_ui = os.path.join(t_path, f"negative_ui_{i}.png")
        if os.path.exists(p_ui): templates.append(cv2.imread(p_ui, 0))
    return [t for t in templates if t is not None]

def get_slot_occupancy(f_range, r_idx, col_idx, buffer_dir, all_files, bg_tpls):
    """
    Scans a window of frames to determine if a slot is EMPTY.
    Uses sliding-window matchTemplate for robustness against coordinate drift.
    """
    y_center = int(ORE0_Y + (r_idx * STEP))
    x_center = int(ORE0_X + (col_idx * STEP))
    
    # 30x30 tight ROI from the grid
    tw, th = 30, 30
    tx, ty = x_center - (tw // 2), y_center - (th // 2)
    
    peak_bg_score = -1.0

    for f_idx in f_range:
        img_gray = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img_gray is None: continue
        
        roi = img_gray[ty : ty + th, tx : tx + tw]
        if roi.shape != (30, 30): continue

        for tpl in bg_tpls:
            # ROI (30x30) slides across Template (48x48)
            res = cv2.matchTemplate(tpl, roi, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(res)[1]
            if score > peak_bg_score:
                peak_bg_score = score
        
        # Optimization: If we found an undeniable background match, stop scanning this slot
        if peak_bg_score >= 0.92:
            break

    # 0 = Empty (High BG match), 1 = Occupied (Low BG match)
    bit = '0' if peak_bg_score >= EMPTY_THRESHOLD else '1'
    return bit, round(float(peak_bg_score), 4)

def process_floor_dna(floor_data, buffer_dir, all_files, bg_tpls):
    """Worker Function: Profiles 24-slot DNA for one floor."""
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    
    results = {'floor_id': f_id, 'start_frame': start_f}
    
    # 1. BOSS DATA ENFORCEMENT
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        special_map = boss.get('special', {}) if boss['tier'] == 'mixed' else None
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            key = f"R{r+1}_S{c}"
            if special_map:
                results[key] = '0' if special_map.get(s_idx) == 'empty' else '1'
            else:
                results[key] = '1' 
            results[f"{key}_score"] = 1.0 
        return results

    # 2. TEMPORAL DNA SCAN
    f_range = range(start_f, min(end_f + 1, start_f + MAX_DNA_WINDOW))
    
    for r_idx in range(4):
        for col in range(6):
            bit, score = get_slot_occupancy(f_range, r_idx, col, buffer_dir, all_files, bg_tpls)
            results[f"R{r_idx+1}_S{col}"] = bit
            results[f"R{r_idx+1}_S{col}_score"] = score

    return results

def run_ore_identification():
    if not os.path.exists(BOUNDARIES_CSV):
        print(f"Error: {BOUNDARIES_CSV} not found.")
        return

    df_floors = pd.read_csv(BOUNDARIES_CSV)
    
    # Apply floor processing limit for faster diagnostic turnaround
    if LIMIT_FLOORS is not None:
        df_floors = df_floors.head(LIMIT_FLOORS)
        print(f"DIAGNOSTIC MODE: Limiting scan to first {LIMIT_FLOORS} floors.")

    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    bg_tpls = load_bg_templates()
    
    print(f"--- STEP 4.1: TEMPORAL DNA PROFILING v1.6 ---")
    print(f"Scanning 24 slots per floor using Robust Sliding-Window detection...")

    inventory = []
    
    # Parallel execution
    worker = partial(process_floor_dna, buffer_dir=buffer_dir, all_files=all_files, bg_tpls=bg_tpls)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            inventory.append(future.result())
            if i % 10 == 0:
                print(f"  Processed {len(inventory)}/{len(df_floors)} floors...", end="\r")

    # 3. ANALYSIS & SAVING
    df_results = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    
    score_cols = [c for c in df_results.columns if '_score' in c]
    bit_cols = [c for c in df_results.columns if c not in score_cols and c not in ['floor_id', 'start_frame']]
    
    df_results[['floor_id', 'start_frame'] + bit_cols].to_csv(OUT_CSV, index=False)
    df_results[['floor_id'] + score_cols].to_csv(DIAG_CSV, index=False)
    
    print(f"\n\n[DONE] DNA Inventory saved to: {OUT_CSV}")
    print(f"Diagnostic Scores saved to: {DIAG_CSV}")

    # SUCCESS SUMMARY
    all_scores = df_results[score_cols].values.flatten()
    empty_scores = all_scores[all_scores >= EMPTY_THRESHOLD]
    occ_scores = all_scores[all_scores < EMPTY_THRESHOLD]
    
    print(f"\n--- DIAGNOSTIC SUMMARY ---")
    print(f"Total slots analyzed: {len(all_scores)}")
    if len(empty_scores) > 0:
        print(f"Mean 'Empty' Match Score: {np.mean(empty_scores):.4f} (Target > 0.85)")
    if len(occ_scores) > 0:
        print(f"Mean 'Occupied' Match Score: {np.mean(occ_scores):.4f} (Target < 0.50)")

    # Visual Proof Generation
    print("\nGenerating DNA visual proofs (Annotated Scores)...")
    for _, row in df_results.iterrows():
        f_id = int(row['floor_id'])
        # Show specific checkpoints for verification
        if f_id % 10 != 0 and f_id not in [1, 5, 25, 50, 75, 99]: continue
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                bit = row[key]
                score = row[f"{key}_score"]
                
                cy = int(ORE0_Y + (r_idx * STEP))
                cx = int(ORE0_X + (col * STEP))
                
                color = (0, 255, 0) if bit == '0' else (0, 0, 255)
                hx, hy = cx + 20, cy + 30
                cv2.putText(img, f"{bit}", (hx-5, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
                cv2.putText(img, f"{bit}", (hx-5, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(img, f"{score:.2f}", (hx-20, hy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
        
        cv2.imwrite(os.path.join(VERIFY_DIR, f"dna_audit_f{f_id:03d}.jpg"), img)

if __name__ == "__main__":
    run_ore_identification()