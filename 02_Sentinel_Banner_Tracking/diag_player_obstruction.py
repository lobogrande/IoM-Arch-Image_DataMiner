# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using the Forensic Trinity:
#          Temporal Consensus with Zonal Isolation to ignore HUD text.
# Version: 4.4 (Static Sprite Optimization & HUD Text Masking)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- 1. GRID & HUD CONSTANTS ---
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

LIMIT_FLOORS = 20 

# --- 2. DATA-DRIVEN DIAGNOSTIC CONSTANTS ---
STATE_COMPLEXITY_THRESHOLD = 500 
# Ores rarely exceed 1800 complexity; values above this are usually HUD text or Player blocks
ANOMALY_COMPLEXITY_THRESHOLD = 1800 
LUMINANCE_SHADOW_FLOOR = 92      
ROTATION_VARIANTS = [-3, 3]

# Gating
PLAYER_PRESENCE_GATE = 0.25  # Set to match the noise floor found in autopsy
HARVEST_COUNT = 15          

BULLY_PENALTIES = {
    'epic1': 0.04, 'epic2': 0.04, 'epic3': 0.05,
    'leg1': 0.06, 'leg2': 0.06, 'leg3': 0.08,
    'myth1': 0.04, 'myth2': 0.05, 'myth3': 0.06,
    'div1': 0.12, 'div2': 0.12, 'div3': 0.15, 
    'com3': 0.04, 'dirt3': 0.04
}

def load_all_templates():
    templates = {'active': {}, 'shadow': {}, 'player': []}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        if "negative_player" in f:
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None: templates['player'].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))
            continue
        if not f.endswith(('.png', '.jpg')) or "_plain_" not in f.lower(): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: templates[state][tier] = []
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        templates[state][tier].append({'img': img_scaled})
    return templates

def check_side_slice_forensics(roi_gray):
    """
    Analyzes variance in two 'safe zones' to confirm background.
    Zones: Far Left [1:8] and Far Right [49:56]
    """
    # Slice 1: Far Left
    left_slice = roi_gray[15:45, 1:8]
    left_std = np.std(left_slice)
    
    # Slice 2: Far Right
    right_slice = roi_gray[15:45, 49:56]
    right_std = np.std(right_slice)
    
    # If EITHER slice is pure background (STD < 13), it's likely an empty slot with a player overlap
    best_std = min(left_std, right_std)
    return best_std, best_std < 13.0

def get_zonal_complexity(roi, is_text_zone=False):
    """Calculates complexity while ignoring the top HUD text area if needed."""
    if is_text_zone:
        # Ignore top 40% where 'Stage XX' text lives
        calculation_area = roi[int(SIDE_PX*0.4):, :]
    else:
        calculation_area = roi
    return cv2.Laplacian(calculation_area, cv2.CV_64F).var()

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_text_zone = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score, best_roi_gray = 0.0, None
    peak_complexity = 0.0
    frame_candidates = []

    for f_idx in f_range:
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img is None: continue
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi.shape != (SIDE_PX, SIDE_PX): continue
        
        # Calculate complexity ignoring HUD text if in top row centers
        comp = get_zonal_complexity(roi, is_text_zone)
        peak_complexity = max(peak_complexity, comp)
        
        # Player Check
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(roi, pt, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        if best_roi_gray is None or comp > get_zonal_complexity(best_roi_gray, is_text_zone):
            best_roi_gray = roi.copy()
            
        # Harvesting: Frame is 'Pristine' if player presence is low and complexity is sane
        if max_p < 0.45 and comp < ANOMALY_COMPLEXITY_THRESHOLD:
            frame_candidates.append(roi)

    # --- FORENSIC TRIGGER ---
    # Force likely_empty call if player is present OR complexity is anomalous (HUD/Player noise)
    if not frame_candidates or peak_complexity > ANOMALY_COMPLEXITY_THRESHOLD:
        if peak_p_score > PLAYER_PRESENCE_GATE or peak_complexity > ANOMALY_COMPLEXITY_THRESHOLD:
            val, is_empty = check_side_slice_forensics(best_roi_gray)
            if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"

    if not frame_candidates:
        return "low_conf", 0, 0, peak_p_score, ""

    # Identification Phase
    tier_momentum = defaultdict(float)
    best_score = 0.0
    for roi in frame_candidates[:HARVEST_COUNT]:
        comp = get_zonal_complexity(roi, is_text_zone)
        target_state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        frame_results = []
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            s = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED))[1] for t in res[target_state][tier]])
            frame_results.append({'tier': tier, 'score': s - penalty})
            
        if frame_results:
            winner = sorted(frame_results, key=lambda x: x['score'], reverse=True)[0]
            tier_momentum[winner['tier']] += 1
            best_score = max(best_score, winner['score'])

    if tier_momentum:
        winner_tier = max(tier_momentum, key=tier_momentum.get)
        gate = 0.40 if 'dirt' in winner_tier or 'com' in winner_tier else 0.48
        if best_score > gate:
            return winner_tier, round(best_score, 4), int(tier_momentum[winner_tier]), peak_p_score, "[M]"

    return "low_conf", round(best_score, 4), 0, peak_p_score, ""

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    if hasattr(cfg, 'BOSS_DATA') and f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_tag"] = "[B]"
        return results

    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = range(int(floor_data['true_start_frame']), int(floor_data['end_frame']) + 1)
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key], results[f"{key}_tag"] = "empty", ""
            else:
                tier, score, momentum, pmax, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_score"], results[f"{key}_mom"], results[f"{key}_pmax"], results[f"{key}_tag"] = tier, score, momentum, pmax, tag
    return results

def run_tier_identification():
    print(f"--- STEP 4.2: TIER IDENTIFICATION v4.4 (HUD/Static Sprite Optimized) ---")
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    buffer_dir, res = cfg.get_buffer_path(0), load_all_templates()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {f_id:03d} processed. [Success: {tag_counts['[Z]']}+{tag_counts['[M]']}, Likely: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty": continue
                color = (0, 255, 255) if tier == "likely_empty" else (0, 0, 255) if tier == "low_conf" else (0, 255, 0)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()