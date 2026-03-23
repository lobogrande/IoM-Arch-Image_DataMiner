# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Step 1 Homing Data 
#          as a hard-gate for player obstruction forensics.
# Version: 4.6 (Homing Supremacy & Background Competition)

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
HOMING_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

LIMIT_FLOORS = 20 # Set to None for full run

# --- 2. LOGIC CONSTANTS ---
Z_TRUST_THRESHOLD = 2.0         # Lowered to recover genuine matches
MIN_ORE_CONFIDENCE = 0.45       # Absolute floor for a match
STATE_COMPLEXITY_THRESHOLD = 550 
ANOMALY_COMPLEXITY_THRESHOLD = 1800 
HARVEST_COUNT = 15          

# --- 3. ASSET LOADERS ---
def load_all_templates():
    templates = {'active': {}, 'shadow': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    
    # Load Backgrounds
    for i in range(10):
        p = os.path.join(t_path, f"background_plain_{i}.png")
        if os.path.exists(p):
            img = cv2.imread(p, 0)
            templates['bg'].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))
    
    # Load Ores
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')) or "_plain_" not in f.lower(): continue
        if any(x in f.lower() for x in ["background", "negative", "player"]): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: templates[state][tier] = []
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        templates[state][tier].append(img_scaled)
    return templates

def check_side_slice_forensics(roi_gray):
    """Forensic check: Is the ground around the player actually empty?"""
    # Slice 1: Far Left [1:9] | Slice 2: Far Right [48:56]
    left_std = np.std(roi_gray[15:45, 1:9])
    right_std = np.std(roi_gray[15:45, 48:56])
    best_std = min(left_std, right_std)
    # If standard deviation is low, the pixels are uniform (Background)
    return best_std, best_std < 14.5

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res, homing_map):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    slot_id = r_idx * 6 + col_idx
    is_text_zone = (r_idx == 0 and col_idx in [2, 3])
    
    frame_results = []
    obstructed_rois = []

    for f_idx in f_range:
        # Check Homing: Is player physically standing here?
        player_is_here = (homing_map.get(f_idx) == slot_id)
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img is None: continue
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi.shape != (SIDE_PX, SIDE_PX): continue
        
        if player_is_here:
            obstructed_rois.append(roi)
            continue

        # Complexity Gating for Text/Noise
        calc_roi = roi[int(SIDE_PX*0.4):, :] if is_text_zone else roi
        comp = cv2.Laplacian(calc_roi, cv2.CV_64F).var()
        
        if comp > ANOMALY_COMPLEXITY_THRESHOLD:
            obstructed_rois.append(roi)
            continue

        # Identification for Pristine Frames
        target_state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        candidates = []
        # 1. Background Competition
        bg_score = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))[1] for t in res['bg']] + [0])
        candidates.append({'tier': 'empty', 'score': bg_score})
        
        # 2. Ore Competition
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            s = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))[1] for t in res[target_state][tier]])
            candidates.append({'tier': tier, 'score': s})
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        frame_results.append(candidates)

    # --- RESOLUTION LOGIC ---
    # Case A: Totally Obstructed (Follow the Homing data)
    if not frame_results and obstructed_rois:
        # Check the best available ROI for background variance
        best_obs_roi = obstructed_rois[0] 
        val, is_empty = check_side_slice_forensics(best_obs_roi)
        if is_empty: return "likely_empty", round(val, 4), 0, "[L]"
        return "obstructed", 0, 0, "[O]"

    # Case B: Temporal Consensus on Pristine Frames
    tier_votes = defaultdict(list)
    for result_set in frame_results[:HARVEST_COUNT]:
        winner = result_set[0]
        # Calculate Z-Score
        scores = [x['score'] for x in result_set]
        mean_s, std_s = np.mean(scores), np.std(scores)
        z = (winner['score'] - mean_s) / max(0.01, std_s)
        
        if z > Z_TRUST_THRESHOLD and winner['score'] > MIN_ORE_CONFIDENCE:
            tier_votes[winner['tier']].append(winner['score'])
        elif winner['tier'] == 'empty' and winner['score'] > 0.85:
            tier_votes['empty'].append(winner['score'])

    if tier_votes:
        winner_tier = max(tier_votes, key=lambda k: len(tier_votes[k]))
        if winner_tier == 'empty': return "empty", 0, 0, ""
        avg_score = np.mean(tier_votes[winner_tier])
        return winner_tier, round(avg_score, 4), len(tier_votes[winner_tier]), "[Z]"

    return "low_conf", 0, 0, ""

def process_floor_tier(floor_data, dna_map, homing_map, buffer_dir, all_files, res):
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
                tier, score, mom, tag = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res, homing_map)
                results[key], results[f"{key}_score"], results[f"{key}_mom"], results[f"{key}_tag"] = tier, score, mom, tag
    return results

def run_tier_identification():
    print(f"--- STEP 4.2: TIER IDENTIFICATION v4.6 (Homing Supremacy) ---")
    if not os.path.exists(HOMING_CSV):
        print(f"Error: {HOMING_CSV} not found.")
        return
        
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    df_homing = pd.read_csv(HOMING_CSV)
    homing_map = df_homing.set_index('frame_idx')['slot_id'].to_dict()
    
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    buffer_dir, res = cfg.get_buffer_path(0), load_all_templates()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    worker = partial(process_floor_tier, dna_map=df_dna, homing_map=homing_map, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {result['floor_id']:03d} processed. [Z-Score: {tag_counts['[Z]']}, LikelyEmpty: {tag_counts['[L]']}, Obstructed: {tag_counts['[O]']}] ({i+1}/{len(df_floors)})")

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
                color = (0, 255, 255) if tier == "likely_empty" else (0, 0, 255) if tier in ["low_conf", "obstructed"] else (0, 255, 0)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()