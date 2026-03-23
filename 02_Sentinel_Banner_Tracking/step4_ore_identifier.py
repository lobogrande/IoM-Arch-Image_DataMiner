# step4_ore_identifier.py
# Purpose: Master Plan Step 4.1 - Establish 100% accurate 24-slot DNA Occupancy
#          using Temporal Window Scanning and Boss-Data Enforcement.
# Version: 1.4.1 (NameError Fix)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_proofs")

# GRID CONSTANTS (Ore Centers)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0

# THRESHOLDS
# Logic: High match with background = Empty (0). 
# If a slot matches BG > 0.75 in ANY frame of the floor, it is Empty.
EMPTY_THRESHOLD = 0.75
MAX_DNA_WINDOW = 200 # Max frames to scan per floor to find an empty window

def load_bg_templates():
    """Loads background and negative UI templates to detect empty space."""
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
    Returns '0' if the background is ever clearly visible, '1' otherwise.
    """
    y_center = int(ORE0_Y + (r_idx * STEP))
    x_center = int(ORE0_X + (col_idx * STEP))
    
    # 30x30 tight crop to avoid grid lines
    tw, th = 30, 30
    tx, ty = x_center - (tw // 2), y_center - (th // 2)
    
    peak_bg_score = -1.0

    for f_idx in f_range:
        img_gray = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img_gray is None: continue
        
        roi = img_gray[ty : ty + th, tx : tx + tw]
        if roi.shape != (30, 30): continue

        for tpl in bg_tpls:
            # Standard template is 48x48, we need 30x30 central crop for DNA
            # Note: Assuming background templates have the same visual features at center
            tpl_crop = tpl[9:39, 9:39] 
            res = cv2.matchTemplate(roi, tpl_crop, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(res)[1]
            if score > peak_bg_score:
                peak_bg_score = score
        
        # Early Exit: If we found a clean background match, the slot is undeniably empty
        if peak_bg_score >= EMPTY_THRESHOLD:
            return '0', round(float(peak_bg_score), 4)

    # If we never saw the background across the whole floor duration, it's occupied
    return '1', round(float(peak_bg_score), 4)

def process_floor_dna(floor_data, buffer_dir, all_files, bg_tpls):
    """Worker Function: Profiles 24-slot DNA for one floor."""
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    
    results = {'floor_id': f_id, 'start_frame': start_f, 'end_frame': end_f}
    
    # 1. BOSS DATA ENFORCEMENT
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        if boss['tier'] != 'mixed':
            # Solid tier boss (e.g. Boss 100 is all Mythic)
            for s_idx in range(24):
                r, c = divmod(s_idx, 6)
                results[f"R{r+1}_S{c}"] = '1'
        else:
            # Mixed tier boss (e.g. Boss 98/99)
            special_map = boss.get('special', {})
            for s_idx in range(24):
                r, c = divmod(s_idx, 6)
                tier = special_map.get(s_idx, 'empty')
                results[f"R{r+1}_S{c}"] = '0' if tier == 'empty' else '1'
        return results

    # 2. TEMPORAL DNA SCAN
    # Fix: Corrected MAX_SCAN_WINDOW to MAX_DNA_WINDOW
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
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    bg_tpls = load_bg_templates()
    
    print(f"--- STEP 4.1: TEMPORAL DNA PROFILING (110 Floors) ---")
    print(f"Scanning 24 slots per floor using Peak Empty detection...")

    inventory = []
    
    # Parallel execution
    worker = partial(process_floor_dna, buffer_dir=buffer_dir, all_files=all_files, bg_tpls=bg_tpls)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            inventory.append(future.result())
            if i % 10 == 0:
                print(f"  Processed {i}/{len(df_floors)} floors...", end="\r")

    # Sort and Save
    df_dna = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    df_dna.to_csv(OUT_CSV, index=False)
    
    print(f"\n\n[DONE] DNA Inventory saved to: {OUT_CSV}")
    
    # Visual Proof Generation (Sample Floors)
    print("Generating DNA visual proofs (Start Frames)...")
    for _, row in df_dna.iterrows():
        f_id = int(row['floor_id'])
        if f_id % 5 != 0: continue
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        
        # Annotate 24 slots
        for r_idx in range(4):
            for col in range(6):
                bit = row[f"R{r_idx+1}_S{col}"]
                cy = int(ORE0_Y + (r_idx * STEP))
                cx = int(ORE0_X + (col * STEP))
                
                color = (0, 255, 0) if bit == '1' else (100, 100, 100)
                # HUD Offsets
                hx, hy = cx + 20, cy + 30
                cv2.putText(img, bit, (hx-5, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3) # Shadow
                cv2.putText(img, bit, (hx-5, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imwrite(os.path.join(VERIFY_DIR, f"dna_audit_f{f_id:03d}.jpg"), img)

if __name__ == "__main__":
    run_ore_identification()