# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Temporal Consensus 
#          and Side-Slice Forensics for permanent player obstructions.
# Version: 1.6 (Progress Telemetry & KeyError Fix)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

# DIAGNOSTIC CONTROL
LIMIT_FLOORS = 20  

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS
MIN_MATCH_CONFIDENCE = 0.45  
PLAYER_REJECTION_GATE = 0.75 
SIDE_SLICE_GATE = 0.85       # Threshold for background match in left 8px slice
HARVEST_COUNT = 15          

def load_resources():
    """Pre-loads ores, players, and background templates."""
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_48 = cv2.resize(img, (48, 48))
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = []
            res['ores'][tier].append(img_48)
        if "negative_player" in f: res['player'].append(img_48)
        if "background_plain" in f: res['bg'].append(img_48)
    return res

def check_side_slice_empty(roi_gray, bg_tpls, is_banner):
    """Forensic check: Does the left side of the slot match the background?"""
    # 30x30 central ore ROI
    roi_30 = roi_gray[13:43, 13:43]
    if is_banner: roi_30 = roi_30[12:, :]
    
    # Peek at the left 8 pixels (most likely to be clear of player sprite)
    slice_roi = roi_30[:, 0:8]
    best_s = 0
    for tpl in bg_tpls:
        # Standard crop for bg is center 30x30
        tpl_30 = tpl[9:39, 9:39]
        if is_banner: tpl_30 = tpl_30[12:, :]
        slice_tpl = tpl_30[:, 0:8]
        res = cv2.matchTemplate(slice_tpl, slice_roi, cv2.TM_CCOEFF_NORMED)
        best_s = max(best_s, cv2.minMaxLoc(res)[1])
    return best_s > SIDE_SLICE_GATE

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score = 0.0
    last_roi_gray = None

    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        last_roi_gray = roi_gray
        
        # Player Obstruction Check
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt, roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        if max_p < PLAYER_REJECTION_GATE:
            quality = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
            frame_candidates.append({'gray': roi_gray, 'quality': quality})

    # Harvest & Tier Identify
    tier_matches = []
    best_overall_score = 0
    top_frames = sorted(frame_candidates, key=lambda x: x['quality'], reverse=True)[:HARVEST_COUNT]
    
    for f in top_frames:
        roi_30 = f['gray'][13:43, 13:43]
        if is_banner: roi_30 = roi_30[12:, :]
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            for tpl in res['ores'][tier]:
                score = cv2.minMaxLoc(cv2.matchTemplate(tpl, roi_30, cv2.TM_CCOEFF_NORMED))[1]
                if score > best_overall_score: best_overall_score = score
                if score >= MIN_MATCH_CONFIDENCE: tier_matches.append(tier)

    if tier_matches:
        winner, win_count = Counter(tier_matches).most_common(1)[0]
        return winner, round(best_overall_score, 4), win_count, peak_p_score

    # FORENSIC FALLBACK: If we only saw the player, check if there's background behind them
    if peak_p_score > PLAYER_REJECTION_GATE and last_roi_gray is not None:
        if check_side_slice_empty(last_roi_gray, res['bg'], is_banner):
            return "likely_empty", 0.0, 0, peak_p_score

    return "low_conf", round(best_overall_score, 4), 0, peak_p_score

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    # FIX: Corrected column name from 'start_frame' to 'true_start_frame'
    start_f = int(floor_data['true_start_frame'])
    results = {'floor_id': f_id, 'start_frame': start_f}
    
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            key = f"R{r+1}_S{c}"
            identity = boss['special'][s_idx] if boss['tier'] == 'mixed' else boss['tier']
            results[key] = identity
            # Consistency: Populate diagnostic columns for bosses
            results[f"{key}_score"] = 1.0
            results[f"{key}_harv"] = 1
            results[f"{key}_pmax"] = 0.0
        return results

    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = range(start_f, int(floor_data['end_frame']) + 1)
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key] = "empty"
                results[f"{key}_score"] = 0.0
                results[f"{key}_harv"] = 0
                results[f"{key}_pmax"] = 0.0
            else:
                tier, score, harv, pmax = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_score"], results[f"{key}_harv"], results[f"{key}_pmax"] = tier, score, harv, pmax
    return results

def run_tier_identification():
    if not os.path.exists(BOUNDARIES_CSV) or not os.path.exists(DNA_INVENTORY_CSV):
        print("Error: Required input CSVs missing.")
        return

    df_floors, df_dna = pd.read_csv(BOUNDARIES_CSV), pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    buffer_dir, res = cfg.get_buffer_path(0), load_resources()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 4.2: TIER IDENTIFICATION v1.6 (Progress Enabled) ---")
    print(f"Parallelizing {len(df_floors)} floors...")
    
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use as_completed for real-time progress logging
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            print(f"  Floor {result['floor_id']:03d} processed ({i+1}/{len(df_floors)})", end="\r")

    final_df = pd.DataFrame(inventory).sort_values('floor_id')
    final_df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Final Ore Inventory saved to: {OUT_CSV}")
    
    print("Generating visual audit proofs...")
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier = str(row[key])
                if tier == "empty": continue
                # Color coding: Green (Success), Yellow (Likely Empty), Red (Fail)
                color = (0, 255, 0) if tier not in ["low_conf", "likely_empty"] else (0, 255, 255) if tier == "likely_empty" else (0, 0, 255)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, tier, (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, tier, (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"Audit images saved to {VERIFY_DIR}")

if __name__ == "__main__":
    run_tier_identification()