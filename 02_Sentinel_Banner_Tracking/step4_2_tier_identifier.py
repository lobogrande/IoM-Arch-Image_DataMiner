# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - High-Accuracy Ore ID using Temporal Variance
#          and Background Competition.
# Version: 5.0 (Variance-Gated Consensus)

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

# --- 2. SURGICAL CONTROLS ---
LIMIT_FLOORS = 20        
MAX_SAMPLES = 40 
MIN_SCORE_GATE = 0.42
STATE_COMPLEXITY_THRESHOLD = 500

def load_all_templates():
    templates = {'active': {}, 'shadow': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    print(f"Loading Resources from {t_path}...")
    
    # Load BG as a tier competitor
    for i in range(10):
        p = os.path.join(t_path, f"background_plain_{i}.png")
        if os.path.exists(p):
            img = cv2.imread(p, 0)
            templates['bg'].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))

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
    # Surgical 2px edge check
    left_std = np.std(roi_gray[20:40, 1:3])
    right_std = np.std(roi_gray[20:40, 54:56])
    best_std = min(left_std, right_std)
    return best_std, best_std < 11.0 # Strict ground signature

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res, homing_map):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2) 
    slot_id = r_idx * 6 + col_idx
    is_text_zone = (r_idx == 0 and col_idx in [2, 3])
    
    # Sampling
    step_size = max(1, len(f_range) // MAX_SAMPLES)
    sample_indices = f_range[::step_size][:MAX_SAMPLES]
    
    rois = []
    for f_idx in sample_indices:
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img is not None:
            roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            if roi.shape == (SIDE_PX, SIDE_PX):
                rois.append({'roi': roi, 'f_idx': f_idx})

    if not rois: return "low_conf", 0, 0, ""

    # 1. Temporal Variance Check (Is the player moving in this slot?)
    stack = np.stack([r['roi'] for r in rois])
    std_map = np.std(stack, axis=0)
    temporal_variance = np.mean(std_map)

    # 2. Obstruction Check via Homing + Variance
    # If variance is very low, the player is static.
    is_static_obstruction = temporal_variance < 3.0
    
    votes = defaultdict(float)
    frames_counted = 0

    for item in rois:
        roi = item['roi']
        f_idx = item['f_idx']
        
        # Skip identification if Step 1 confirms player is standing here
        if homing_map.get(f_idx) == slot_id: continue

        calc_roi = roi[int(SIDE_PX*0.4):, :] if is_text_zone else roi
        comp = cv2.Laplacian(calc_roi, cv2.CV_64F).var()
        
        target_state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        candidates = []
        # Competitor: Background
        bg_score = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))[1] for t in res['bg']])
        candidates.append({'tier': 'background', 'score': bg_score})
        
        # Competitor: Ores
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            score = max([cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))[1] for t in res[target_state][tier]])
            candidates.append({'tier': tier, 'score': score})
            
        winner = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
        if winner['score'] > MIN_SCORE_GATE:
            votes[winner['tier']] += winner['score']
            frames_counted += 1

    # Decision Logic
    if frames_counted == 0:
        # We are totally obstructed. Run forensics on the median frame.
        median_roi = np.median(stack, axis=0).astype(np.uint8)
        val, is_empty = check_side_slice_forensics(median_roi)
        if is_empty: return "likely_empty", round(val, 4), len(rois), "[L]"
        else: return "unknown_obstructed", 0, len(rois), "[U]"

    if votes:
        winning_tier = max(votes, key=votes.get)
        if winning_tier == 'background':
            return "empty_consensus", 0, frames_counted, "[L]"
        return winning_tier, round(votes[winning_tier]/frames_counted, 4), frames_counted, "[M]"

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
    print(f"--- STEP 4.2: TIER IDENTIFICATION v5.0 (Variance-Gated) ---")
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
    
    total = len(df_floors)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            result = future.result()
            inventory.append(result)
            tags = Counter([v for k, v in result.items() if k.endswith('_tag')])
            print(f"  ({count}/{total}) Floor {result['floor_id']:03d} | Cyan: {tags['[L]']} | Yellow: {tags['[U]']}")

    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty" or tier == "background": continue
                color = (255, 255, 0) if tag == "[L]" else (0, 255, 255) if tag == "[U]" else (0, 0, 255) if tier == "low_conf" else (0, 255, 0)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.35, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.35, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()