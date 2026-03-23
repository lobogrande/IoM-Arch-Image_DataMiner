# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers for occupied slots using 
#          Temporal Consensus and DNA Ingestion.
# Version: 1.0 (Standalone Tier Identification)

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

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS
MIN_ID_CONFIDENCE = 0.52   
MAX_HARVEST_FRAMES = 30   
COMPLEXITY_GATE = 350      

def load_ore_templates():
    res = {'ores': {}, 'mask': None}
    t_path = cfg.TEMPLATE_DIR
    mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(mask, (SIDE_PX//2, SIDE_PX//2), int(18 * SCALE), 255, -1)
    res['mask'] = mask
    for f in os.listdir(t_path):
        if "_plain_" in f and "_act_" in f and not any(x in f for x in ["player", "negative", "background"]):
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None:
                tier = f.split("_")[0]
                if tier not in res['ores']: res['ores'][tier] = []
                res['ores'][tier].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))
    return res

def is_pristine(roi_bgr, roi_gray):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    if cv2.countNonZero(mask) > 120: return False
    if cv2.Laplacian(roi_gray, cv2.CV_64F).var() < COMPLEXITY_GATE: return False
    return True

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    harvested_matches = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        if is_pristine(roi_bgr, roi_gray):
            best_s, best_t = -1, None
            for tier in allowed_tiers:
                if tier not in res['ores']: continue
                for tpl in res['ores'][tier]:
                    score = cv2.minMaxLoc(cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED, mask=res['mask']))[1]
                    if score > best_s: best_s, best_t = score, tier
            if best_t and best_s >= MIN_ID_CONFIDENCE: harvested_matches.append(best_t)
        if len(harvested_matches) >= MAX_HARVEST_FRAMES: break
    if not harvested_matches: return "low_conf", 0.0
    winner, win_count = Counter(harvested_matches).most_common(1)[0]
    return winner, round(win_count / len(harvested_matches), 2)

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    results = {'floor_id': f_id, 'start_frame': start_f}
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            results[f"R{r+1}_S{c}"] = boss['special'][s_idx] if boss['tier'] == 'mixed' else boss['tier']
        return results
    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0': results[key] = "empty"
            else:
                tier, conf = identify_consensus(range(start_f, end_f+1), r_idx, col, buffer_dir, all_files, allowed, res)
                results[key], results[f"{key}_conf"] = tier, conf
    return results

def run_tier_identification():
    if not os.path.exists(DNA_INVENTORY_CSV): return
    df_floors, df_dna = pd.read_csv(BOUNDARIES_CSV), pd.read_csv(DNA_INVENTORY_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    res = load_ore_templates()
    print(f"--- STEP 4.2: TIER IDENTIFICATION ---")
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        inventory = list(executor.map(worker, [r for _, r in df_floors.iterrows()]))
    pd.DataFrame(inventory).sort_values('floor_id').to_csv(OUT_CSV, index=False)
    print(f"[DONE] Final Ore Inventory saved to: {OUT_CSV}")

if __name__ == "__main__":
    run_tier_identification()