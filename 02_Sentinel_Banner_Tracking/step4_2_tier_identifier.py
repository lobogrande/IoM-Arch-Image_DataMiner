# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Temporal Consensus,
#          Side-Slice Forensics for player overlap, and Game-Rule Enforcement.
# Version: 1.8 (The Forensic Pure-Ore Identifier)

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
LIMIT_FLOORS = 20  # Set to None for production; use small numbers for testing

# GRID CONSTANTS (VERIFIED)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS
MIN_MATCH_CONFIDENCE = 0.45  # Direct identification gate
PROMOTION_THRESHOLD = 0.35   # Lowered floor for Winner-Take-All consensus
PLAYER_REJECTION_GATE = 0.75 
SIDE_SLICE_GATE = 0.70       # Calibrated for background peek (10px strip)
HARVEST_COUNT = 15          

def load_resources():
    """Loads ore tiers, player sprites, and background (isolated for forensics)."""
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_48 = cv2.resize(img, (48, 48))
        
        # 1. Pure Ore Tiers (Main Search)
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = []
            res['ores'][tier].append(img_48)
            
        # 2. Player Sprites (Obstruction Check)
        if "negative_player" in f: 
            res['player'].append(img_48)
            
        # 3. Background Templates (Restricted to Side-Slice Forensics)
        if "background_plain" in f: 
            res['bg'].append(img_48)
            
    return res

def check_side_slice_empty(roi_gray, bg_tpls, is_banner):
    """Forensic check: Peeks at the left 10px to see background behind player."""
    roi_30 = roi_gray[13:43, 13:43]
    if is_banner: roi_30 = roi_30[12:, :]
    
    slice_roi = roi_30[:, 0:10]
    best_s = 0
    for tpl in bg_tpls:
        tpl_30 = tpl[9:39, 9:39]
        if is_banner: tpl_30 = tpl_30[12:, :]
        slice_tpl = tpl_30[:, 0:10]
        res = cv2.matchTemplate(slice_tpl, slice_roi, cv2.TM_CCOEFF_NORMED)
        best_s = max(best_s, cv2.minMaxLoc(res)[1])
    return best_s, best_s > SIDE_SLICE_GATE

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Temporal consensus search prioritizing clean (unobstructed) frames."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score = 0.0
    last_roi_gray = None

    # Phase 1: Harvesting
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_gray = cv2.cvtColor(img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX], cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        last_roi_gray = roi_gray
        
        # Player Check (30x30 core)
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt, roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        if max_p < PLAYER_REJECTION_GATE:
            quality = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
            frame_candidates.append({'gray': roi_gray, 'quality': quality})

    # Phase 2: Identification
    tier_tallies = []
    valid_matches = []
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
                if score >= MIN_MATCH_CONFIDENCE: valid_matches.append(tier)
                tier_tallies.append({'tier': tier, 'score': score})

    # Phase 3: Resolution
    if valid_matches:
        winner, count = Counter(valid_matches).most_common(1)[0]
        return winner, round(best_overall_score, 4), count, peak_p_score, ""

    if tier_tallies:
        # Winner-Take-All Promotion for low-tier ores
        counts = Counter([t['tier'] for t in sorted(tier_tallies, key=lambda x: x['score'], reverse=True)[:HARVEST_COUNT*3]])
        winner, count = counts.most_common(1)[0]
        max_win_score = max([t['score'] for t in tier_tallies if t['tier'] == winner])
        if max_win_score >= PROMOTION_THRESHOLD:
            return winner, round(max_win_score, 4), count, peak_p_score, "[P]"

    # Phase 4: Forensic Fallback (Likely Empty)
    if peak_p_score > PLAYER_REJECTION_GATE and last_roi_gray is not None:
        slice_s, is_empty = check_side_slice_empty(last_roi_gray, res['bg'], is_banner)
        if is_empty: return "likely_empty", round(slice_s, 4), 0, peak_p_score, "[L]"

    return "low_conf", round(best_overall_score, 4), 0, peak_p_score, ""

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    # 1. UNABRIDGED BOSS DATA ENFORCEMENT
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            results[f"R{r+1}_S{c}"] = boss['special'][s_idx] if boss['tier'] == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}_tag"] = ""
        return results

    # 2. ORE RESTRICTION FILTERING
    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = range(int(floor_data['true_start_frame']), int(floor_data['end_frame']) + 1)
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key], results[f"{key}_tag"] = "empty", ""
            else:
                tier, score, harv, pmax, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_score"], results[f"{key}_harv"], results[f"{key}_pmax"], results[f"{key}_tag"] = tier, score, harv, pmax, tag
    return results

def run_tier_identification():
    if not os.path.exists(DNA_INVENTORY_CSV): return
    df_floors, df_dna = pd.read_csv(BOUNDARIES_CSV), pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    buffer_dir, res = cfg.get_buffer_path(0), load_resources()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 4.2: TIER IDENTIFICATION v1.8 (Forensic Pure-Ore) ---")
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            stats = Counter([v for k, v in result.items() if k.startswith('R') and '_' not in k])
            print(f"  Floor {result['floor_id']:03d}: LowConf: {stats['low_conf']}, LikelyEmpty: {sum(1 for k,v in result.items() if v == '[L]')} ({i+1}/{len(df_floors)})", end="\r")

    final_df = pd.DataFrame(inventory).sort_values('floor_id')
    final_df.to_csv(OUT_CSV, index=False)
    
    print("\nGenerating visual audit proofs...")
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty": continue
                color = (0, 255, 0) if tier not in ["low_conf", "likely_empty"] else (0, 255, 255) if tier == "likely_empty" else (0, 0, 255)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx+HUD_DX-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()