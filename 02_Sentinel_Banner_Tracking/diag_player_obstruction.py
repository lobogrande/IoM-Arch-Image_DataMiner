# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Step 1 Homing Data 
#          integration and Z-Score Conflict Rejection.
# Version: 4.5 (Homing Integration & Bully Shield)

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

# --- 2. DATA-DRIVEN DIAGNOSTIC CONSTANTS ---
STATE_COMPLEXITY_THRESHOLD = 500 
ANOMALY_COMPLEXITY_THRESHOLD = 1800 
Z_TRUST_THRESHOLD = 2.4 # Higher = Stricter Bully Protection

BULLY_PENALTIES = {
    'epic1': 0.05, 'epic2': 0.05, 'epic3': 0.06,
    'leg1': 0.07, 'leg2': 0.07, 'leg3': 0.09,
    'myth1': 0.05, 'myth2': 0.06, 'myth3': 0.07,
    'div1': 0.15, 'div2': 0.15, 'div3': 0.20, 
    'com3': 0.05, 'dirt3': 0.05
}

def load_all_templates():
    templates = {'active': {}, 'shadow': {}}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')) or "_plain_" not in f.lower(): continue
        if any(x in f.lower() for x in ["background", "negative", "player"]): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: templates[state][tier] = []
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        templates[state][tier].append({'img': img_scaled})
    return templates

def check_side_slice_forensics(roi_gray):
    """Checks far-left and far-right slivers for background signature."""
    left_slice = roi_gray[15:45, 1:9]
    right_slice = roi_gray[15:45, 48:56]
    best_std = min(np.std(left_slice), np.std(right_slice))
    return best_std, best_std < 14.0

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res, homing_map):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    slot_id = r_idx * 6 + col_idx
    is_text_zone = (r_idx == 0 and col_idx in [2, 3])
    
    pristine_candidates = []
    obstructed_best_roi = None
    
    for f_idx in f_range:
        # Check Step 1 Homing Data: Is player physically standing here?
        player_at_slot = homing_map.get(f_idx)
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img is None: continue
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi.shape != (SIDE_PX, SIDE_PX): continue
        
        # Calculate complexity (ignoring HUD text if applicable)
        calc_roi = roi[int(SIDE_PX*0.4):, :] if is_text_zone else roi
        comp = cv2.Laplacian(calc_roi, cv2.CV_64F).var()

        if player_at_slot == slot_id or comp > ANOMALY_COMPLEXITY_THRESHOLD:
            if obstructed_best_roi is None or comp > cv2.Laplacian(obstructed_best_roi, cv2.CV_64F).var():
                obstructed_best_roi = roi.copy()
        else:
            pristine_candidates.append({'roi': roi, 'comp': comp})

    # --- FORENSIC FALLBACK: If mostly/totally obstructed ---
    if len(pristine_candidates) < 3:
        if obstructed_best_roi is not None:
            val, is_empty = check_side_slice_forensics(obstructed_best_roi)
            if is_empty: return "likely_empty", round(val, 4), 0, "[L]"
        return "low_conf", 0, 0, ""

    # --- IDENTIFICATION: Use Pristine (Player-Free) Frames ---
    tier_scores = defaultdict(list)
    for cand in pristine_candidates[:15]:
        roi = cand['roi']
        target_state = 'active' if cand['comp'] > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        frame_results = []
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            penalty = BULLY_PENALTIES.get(tier, 0.0)
            s = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED))[1] for t in res[target_state][tier]])
            frame_results.append({'tier': tier, 'score': s - penalty})
            
        if frame_results:
            # Sort by score for Z-Score calculation
            frame_results.sort(key=lambda x: x['score'], reverse=True)
            winner = frame_results[0]
            
            # Bully Shield: Z-Score against the field
            all_scores = [x['score'] for x in frame_results]
            mean_s, std_s = np.mean(all_scores), np.std(all_scores)
            z = (winner['score'] - mean_s) / max(0.01, std_s)
            
            if z > Z_TRUST_THRESHOLD:
                tier_scores[winner['tier']].append(winner['score'])

    if tier_scores:
        winner_tier = max(tier_scores, key=lambda k: len(tier_scores[k]))
        avg_score = np.mean(tier_scores[winner_tier])
        return winner_tier, round(avg_score, 4), len(tier_scores[winner_tier]), "[Z]"

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
    print(f"--- STEP 4.2: TIER IDENTIFICATION v4.5 (Homing-Integrated) ---")
    if not os.path.exists(DNA_INVENTORY_CSV) or not os.path.exists(HOMING_CSV):
        print("Error: DNA Inventory or Homing CSV missing.")
        return
        
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    df_homing = pd.read_csv(HOMING_CSV)
    homing_map = df_homing.set_index('frame_idx')['slot_id'].to_dict()
    
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
        
    buffer_dir = cfg.get_buffer_path(0)
    res = load_all_templates()
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    worker = partial(process_floor_tier, dna_map=df_dna, homing_map=homing_map, buffer_dir=buffer_dir, all_files=all_files, res=res)
    inventory = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            tag_counts = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  Floor {f_id:03d} processed. [Z-Score: {tag_counts['[Z]']}, LikelyEmpty: {tag_counts['[L]']}] ({i+1}/{len(df_floors)})")

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