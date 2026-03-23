# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - High-Accuracy Ore ID using Row-Aware 
#          Overlap Discrimination and Homing-Locked Forensics.
# Version: 5.3 (Row-Boundary Enforcement & Directional Logic)

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

# Test Control
LIMIT_FLOORS = 20        

# Logic Constants
MAX_SAMPLES = 60         
MIN_SCORE_GATE = 0.40
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
        img_scaled = cv2.resize(img_raw, (SIDE_PX, SIDE_PX), interpolation=cv2.INTER_AREA)
        for angle in ROTATION_VARIANTS:
            templates[state][tier].append(rotate_image(img_scaled, angle))
    return templates

def check_side_slice_forensics(roi_gray):
    """Surgical 2px sliver check of the left edge to confirm background ground."""
    left_slice = roi_gray[20:40, 1:3]
    right_slice = roi_gray[20:40, 54:56]
    best_std = min(np.std(left_slice), np.std(right_slice))
    return best_std, best_std < 11.5

def get_overlap_slot(homing_id):
    """
    Directional Overlap Logic:
    Facing Right (Default): Overlaps slot to the Left (N-1).
    Facing Left (Slot 11): Overlaps slot to the Right (N+1).
    """
    if homing_id is None or homing_id < 0: return -99
    
    row = homing_id // 6
    # Identify direction based on Step 1 configuration
    is_facing_left = (homing_id == 11)
    
    overlap_candidate = (homing_id + 1) if is_facing_left else (homing_id - 1)
    
    # HARD GATE: The overlapped area MUST be in the same row and within the grid
    if overlap_candidate // 6 != row or not (0 <= overlap_candidate <= 23):
        return -99 # Overlap falls off the grid/row
        
    return overlap_candidate

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res, homing_map):
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2) 
    slot_id = r_idx * 6 + col_idx
    is_text_zone = (r_idx == 0 and col_idx in [2, 3])
    
    # 1. PERMANENT OCCLUSION GATING
    total_frames = len(f_range)
    # Filter frames where the player is PHYSICALLY OVERLAPPING this specific slot
    frames_overlapping = [f for f in f_range if get_overlap_slot(homing_map.get(f)) == slot_id]
    
    if len(frames_overlapping) == total_frames:
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_range[0]]), 0)
        if img is not None:
            roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            val, is_empty = check_side_slice_forensics(roi)
            if is_empty: return "likely_empty", round(val, 4), total_frames, "[L]"
            else: return "obstructed", 0, total_frames, "[O]"

    # 2. CLEAN VOTING PATH
    # We sample indices where the player is NOT overlapping the slot
    clean_indices = [f for f in f_range if get_overlap_slot(homing_map.get(f)) != slot_id]
    if not clean_indices: clean_indices = f_range 
    
    if len(clean_indices) > MAX_SAMPLES:
        step = len(clean_indices) // MAX_SAMPLES
        clean_indices = clean_indices[::step][:MAX_SAMPLES]

    votes = defaultdict(float)
    for f_idx in clean_indices:
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        if img is None: continue
        roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        
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
        
        if best_f_tier and best_f_score > MIN_SCORE_GATE:
            votes[best_f_tier] += best_f_score

    if votes:
        winner = max(votes, key=votes.get)
        return winner, round(votes[winner]/len(clean_indices), 4), len(clean_indices), "[M]"

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
    print(f"--- STEP 4.2: TIER IDENTIFICATION v5.3 (Row-Boundary Enforcement) ---")
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
            print(f"  ({count}/{total}) Floor {result['floor_id']:03d} | Cyan: {tags['[L]']} | Yellow: {tags['[O]']}")

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
                
                # Colors: BGR
                if tag == "[L]": color = (255, 255, 0)      # Cyan
                elif tag == "[O]": color = (0, 255, 255)    # Yellow
                elif tier == "low_conf": color = (0, 0, 255) # Red
                else: color = (0, 255, 0)                   # Green
                
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.35, (0,0,0), 2)
                cv2.putText(img, f"{tier}{tag}", (cx-25, cy+HUD_DY), 0, 0.35, color, 1)
        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img)
    print(f"[DONE] Final Inventory: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()