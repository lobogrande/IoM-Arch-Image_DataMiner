# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - High-Accuracy Ore ID using Validated Grid Anchor.
# Version: 5.5 (Corrected Slot-Offset & Forensic Reporting)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- 1. VALIDATED PIXEL CONSTANTS ---
ORE0_X, ORE0_Y = 74, 261  # Corrected Anchor (133 - 59)
STEP = 59.0
SIDE_PX = 48 
HUD_DX, HUD_DY = 20, 30

BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
HOMING_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

# Test Set Control
LIMIT_FLOORS = 20 

# Logic Constants
MAX_SAMPLES = 40         # Focused arrival window
MIN_VOTE_CONFIDENCE = 0.45 
STATE_COMPLEXITY_THRESHOLD = 500
ROTATION_VARIANTS = [-3, 0, 3]

def rotate_image(image, angle):
    if angle == 0: return image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

def load_all_templates():
    templates = {'active': {}, 'shadow': {}}
    t_path = cfg.TEMPLATE_DIR
    print(f"Loading Resources from {t_path}...")
    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')) or "_plain_" not in f.lower(): continue
        if any(x in f.lower() for x in ["background", "negative", "player"]): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        if tier not in templates[state]: templates[state][tier] = []
        img_native = cv2.resize(img_raw, (SIDE_PX, SIDE_PX))
        for angle in ROTATION_VARIANTS:
            templates[state][tier].append(rotate_image(img_native, angle))
    return templates

def check_side_slice_forensics(roi_gray):
    """Surgical 2px edge check to differentiate empty ground from obscured ore."""
    left_slice = roi_gray[15:40, 1:3]
    right_slice = roi_gray[15:40, 45:47]
    best_std = min(np.std(left_slice), np.std(right_slice))
    return best_std, best_std < 11.5

def get_overlap_slot(homing_id):
    """Calculates which slot the player is physically covering."""
    if homing_id is None or homing_id < 0: return -99
    row = homing_id // 6
    # Left overlap if facing left (Slot 11), otherwise Right overlap
    overlap_candidate = (homing_id + 1) if homing_id == 11 else (homing_id - 1)
    if overlap_candidate // 6 != row or not (0 <= overlap_candidate <= 23):
        return -99
    return overlap_candidate

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res, homing_map):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2) 
    slot_id = r_idx * 6 + col_idx
    is_text_zone = (r_idx == 0 and col_idx in [2, 3])
    
    # Slice the sampling to the arrival window (earliest available frames)
    sample_indices = f_range[:MAX_SAMPLES]
    
    votes = defaultdict(float)
    frames_obstructed = 0
    obstructed_sample_roi = None

    for f_idx in sample_indices:
        # Check if the player is physically on TOP of this slot
        player_is_blocking = (get_overlap_slot(homing_map.get(f_idx)) == slot_id)
        
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img is None: continue
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi.shape != (SIDE_PX, SIDE_PX): continue

        if player_is_blocking:
            frames_obstructed += 1
            if obstructed_sample_roi is None: obstructed_sample_roi = roi.copy()
            continue

        # Complexity check for text/noise suppression
        calc_roi = roi[int(SIDE_PX*0.4):, :] if is_text_zone else roi
        comp = cv2.Laplacian(calc_roi, cv2.CV_64F).var()
        target_state = 'active' if comp > STATE_COMPLEXITY_THRESHOLD else 'shadow'
        
        best_f_tier, best_f_score = None, 0
        for tier in allowed_tiers:
            if tier not in res[target_state]: continue
            for tpl in res[target_state][tier]:
                score = cv2.minMaxLoc(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED))[1]
                if score > best_f_score:
                    best_f_score = score
                    best_f_tier = tier
        
        if best_f_tier and best_f_score > MIN_VOTE_CONFIDENCE:
            votes[best_f_tier] += best_f_score

    # --- FORENSIC RESOLUTION (100% Occlusion Cases) ---
    if frames_obstructed == len(sample_indices) and obstructed_sample_roi is not None:
        val, is_empty = check_side_slice_forensics(obstructed_sample_roi)
        if is_empty: return "likely_empty", round(val, 4), frames_obstructed, "[L]"
        else: return "unknown_obstructed", 0, frames_obstructed, "[U]"

    # --- CONSENSUS RESOLUTION ---
    if votes:
        winner = max(votes, key=votes.get)
        return winner, round(votes[winner]/len(sample_indices), 4), len(sample_indices), "[M]"

    return "low_conf", 0, 0, ""

def process_floor_tier(floor_data, dna_map, homing_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    results = {'floor_id': f_id, 'start_frame': int(floor_data['true_start_frame'])}
    
    if hasattr(cfg, 'BOSS_DATA') and f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            results[f"R{r+1}_S{c}"] = boss['special'][s_idx] if boss.get('tier') == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}_tag"] = "[B]"
        return results

    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    f_range = list(range(int(floor_data['true_start_frame']), int(floor_data['end_frame']) + 1))
    
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
    print(f"--- STEP 4.2: TIER IDENTIFICATION v5.5 (Validated Offset) ---")
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    df_homing = pd.read_csv(HOMING_CSV)
    homing_map = df_homing.set_index('frame_idx')['slot_id'].to_dict()
    df_floors = pd.read_csv(BOUNDARIES_CSV)
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
    
    # Generate Visual Proofs
    for _, row in final_df.iterrows():
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier, tag = str(row[key]), str(row.get(f"{key}_tag", ""))
                if tier == "empty": continue
                
                # Colors: BGR Format
                if tag == "[L]": color = (255, 255, 0)      # Cyan
                elif tag == "[U]": color = (0, 255, 255)    # Yellow
                elif tier == "low_conf": color = (0, 0, 255) # Red
                else: color = (0, 255, 0)                   # Green (Consensus)
                
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.4, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()