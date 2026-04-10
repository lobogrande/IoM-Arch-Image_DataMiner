# step5_floor_occupancy.py
# Purpose: Master Plan Step 5 - Establish accurate 24-slot DNA Occupancy 
#          per floor to act as a mask for tier identification.
# Version: 2.1 (Progress Tracking & Visual Proofs)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg
from ascension_detector import auto_configure_ascension

# --- DYNAMIC CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path()
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"final_floor_boundaries_run_{RUN_ID}.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_dna_inventory_run_{RUN_ID}.csv")
DIAG_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"dna_score_analysis_run_{RUN_ID}.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_dna_proofs_run_{RUN_ID}")

# --- VALIDATED CONSTANTS ---
ORE0_X, ORE0_Y = 74, 261
STEP = 59.0
SIDE_PX = 48

# DIAGNOSTIC CONTROL
LIMIT_FLOORS = None  # Process all floors

# THRESHOLDS
EMPTY_THRESHOLD = 0.80 
MAX_DNA_WINDOW = 150  # Keep long to allow UI elements to move away
EARLY_STOP_OCCUPIED = 0.40  # Stop scanning when clear block detected 

def load_bg_templates():
    """Loads background and negative UI templates only. Player is excluded."""
    templates =[]
    t_path = cfg.TEMPLATE_DIR
    for i in range(10):
        p_bg = os.path.join(t_path, f"background_plain_{i}.png")
        if os.path.exists(p_bg): templates.append(cv2.imread(p_bg, 0))
        p_ui = os.path.join(t_path, f"negative_ui_{i}.png")
        if os.path.exists(p_ui): templates.append(cv2.imread(p_ui, 0))
    return [t for t in templates if t is not None]

def get_slot_occupancy(f_range, r_idx, col_idx, buffer_dir, all_files, bg_tpls):
    y_center = int(ORE0_Y + (r_idx * STEP))
    x_center = int(ORE0_X + (col_idx * STEP))
    tw, th = 30, 30
    tx, ty = x_center - (tw // 2), y_center - (th // 2)
    
    is_banner_slot = (r_idx == 0 and col_idx in [2, 3])
    peak_bg_score = -1.0

    for f_idx in f_range:
        img_gray = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img_gray is None: continue
        roi = img_gray[ty : ty + th, tx : tx + tw]
        if roi.shape != (30, 30): continue

        roi_proc = roi[12:, :] if is_banner_slot else roi

        for tpl in bg_tpls:
            # We use normalized cross-correlation for background stability
            res = cv2.matchTemplate(tpl, roi_proc, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(res)[1]
            if score > peak_bg_score:
                peak_bg_score = score
        
        # Early exit if we find a clear block (low background score)
        if peak_bg_score <= EARLY_STOP_OCCUPIED:
            break
        
        # Early exit if we find a perfect background match
        if peak_bg_score >= 0.96: break

    # Logic: If it doesn't match background, it's Occupied (1). 
    bit = '0' if peak_bg_score >= EMPTY_THRESHOLD else '1'
    return bit, round(float(peak_bg_score), 4)

def process_floor_dna(floor_data, buffer_dir, all_files, bg_tpls):
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    results = {'floor_id': f_id, 'start_frame': start_f}
    
    # Bypass for Boss Floors (Hardcoded in project_config)
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        special_map = boss.get('special', {}) if boss['tier'] == 'mixed' else None
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            key = f"R{r+1}_S{c}"
            # Mark as occupied (1) unless explicitly empty in special mixed boss data
            results[key] = ('0' if (special_map and special_map.get(s_idx) == 'empty') else '1')
            results[f"{key}_score"] = 1.0 
        return results

    # Scan a window at the start of the floor to determine initial occupancy
    f_range = range(start_f, min(end_f + 1, start_f + MAX_DNA_WINDOW))
    for r_idx in range(4):
        for col in range(6):
            bit, score = get_slot_occupancy(f_range, r_idx, col, buffer_dir, all_files, bg_tpls)
            results[f"R{r_idx+1}_S{col}"] = bit
            results[f"R{r_idx+1}_S{col}_score"] = score
    return results

def run_dna_profiling():
    auto_configure_ascension()
    print(f"--- STEP 5: FLOOR OCCUPANCY PROFILING (Run {RUN_ID}) ---")
    if not os.path.exists(BOUNDARIES_CSV):
        print(f"Error: {os.path.basename(BOUNDARIES_CSV)} not found. Run Step 4 first.")
        return

    df_floors = pd.read_csv(BOUNDARIES_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    all_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    bg_tpls = load_bg_templates()
    
    worker = partial(process_floor_dna, buffer_dir=SOURCE_DIR, all_files=all_files, bg_tpls=bg_tpls)
    inventory =[]
    
    total = len(df_floors)
    print(f"Sampling temporal window for {total} floors...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            result = future.result()
            inventory.append(result)
            print(f"  Processed ({count}/{total}) Floor {result['floor_id']:03d}", end="\r")

    df_results = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    
    # Column Organization
    score_cols = [c for c in df_results.columns if '_score' in c]
    bit_cols = [c for c in df_results.columns if c not in score_cols and c not in['floor_id', 'start_frame']]
    
    # Save Data
    df_results[['floor_id', 'start_frame'] + bit_cols].to_csv(OUT_CSV, index=False)
    df_results[['floor_id'] + score_cols].to_csv(DIAG_CSV, index=False)
    print(f"\n[DONE] Occupancy Mask saved to: {os.path.basename(OUT_CSV)}")

    # --- GENERATE VISUAL PROOFS ---
    print("Generating Visual DNA Proofs...")
    for _, row in df_results.iterrows():
        f_id = int(row['floor_id'])
        start_f = int(row['start_frame'])
        
        img = cv2.imread(os.path.join(SOURCE_DIR, all_files[start_f]))
        if img is None: continue
        
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                bit = str(row[key])
                
                cy = int(ORE0_Y + (r_idx * STEP))
                cx = int(ORE0_X + (col * STEP))
                x1, y1 = cx - (SIDE_PX // 2), cy - (SIDE_PX // 2)
                
                # Color logic: Green for Occupied (1), Red for Empty (0)
                color = (0, 255, 0) if bit == '1' else (0, 0, 255)
                
                # Draw Box and Bit
                cv2.rectangle(img, (x1, y1), (x1 + SIDE_PX, y1 + SIDE_PX), color, 1)
                cv2.putText(img, bit, (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(img, bit, (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
        out_name = f"dna_proof_f{f_id:03d}.jpg"
        cv2.imwrite(os.path.join(VERIFY_DIR, out_name), img)
        
    print(f"[DONE] Visual proofs saved to {os.path.basename(VERIFY_DIR)}")

if __name__ == "__main__":
    run_dna_profiling()