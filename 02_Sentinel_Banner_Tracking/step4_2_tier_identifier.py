# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers for occupied slots using 
#          Adaptive Temporal Harvesting, Banner Slicing, and Winner-Take-All Fallback.
# Version: 1.3 (The Winner-Take-All Scalpel)

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
MIN_MATCH_CONFIDENCE = 0.45  # Hard gate for automatic acceptance
PROMOTION_THRESHOLD = 0.38   # Minimum score to even consider for Winner-Take-All
HARVEST_COUNT = 15          

def load_ore_templates():
    """Pre-loads pristine active templates (48x48) into memory."""
    res = {'ores': {}}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        if "_plain_" in f and "_act_" in f and not any(x in f for x in ["player", "negative", "background"]):
            img_raw = cv2.imread(os.path.join(t_path, f), 0)
            if img_raw is not None:
                tier = f.split("_")[0]
                if tier not in res['ores']: res['ores'][tier] = []
                res['ores'][tier].append(cv2.resize(img_raw, (48, 48)))
    return res

def get_frame_quality(roi_bgr, roi_gray):
    """Calculates a quality score for a frame (Higher is cleaner)."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    ui_pixels = cv2.countNonZero(mask)
    ui_penalty = ui_pixels * 0.5
    complexity = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    return complexity - ui_penalty

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Grades all frames, picks the best ones, and votes on identity with fallback logic."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner_slot = (r_idx == 0 and col_idx in [2, 3])

    # Phase 1: Quality Grading
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        quality = get_frame_quality(roi_bgr, roi_gray)
        frame_candidates.append({'idx': f_idx, 'quality': quality, 'gray': roi_gray})

    if not frame_candidates:
        return "low_conf", 0.0, 0, 0.0

    # Phase 2: Harvest the Top N Frames
    top_frames = sorted(frame_candidates, key=lambda x: x['quality'], reverse=True)[:HARVEST_COUNT]
    
    # Phase 3: Identify Tiers
    tier_tallies = [] # Track every "Best Match" per frame
    valid_matches = [] # Only those above hard gate
    total_score = 0.0
    
    roi_side = 30
    offset = (SIDE_PX - roi_side) // 2

    for frame in top_frames:
        roi_30 = frame['gray'][offset:offset+roi_side, offset:offset+roi_side]
        # UI MITIGATION: Physical Slicing for Banner Slots
        if is_banner_slot: roi_30 = roi_30[12:, :]

        best_f_score = -1
        best_f_tier = None
        
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            for tpl in res['ores'][tier]:
                match_res = cv2.matchTemplate(tpl, roi_30, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(match_res)[1]
                if score > best_f_score:
                    best_f_score = score
                    best_f_tier = tier
        
        if best_f_tier:
            tier_tallies.append({'tier': best_f_tier, 'score': best_f_score})
            if best_f_score >= MIN_MATCH_CONFIDENCE:
                valid_matches.append(best_f_tier)
                total_score += best_f_score

    avg_quality = np.mean([f['quality'] for f in top_frames])

    # Case A: Clear Consensus above hard gate
    if valid_matches:
        winner, win_count = Counter(valid_matches).most_common(1)[0]
        avg_score = total_score / len(valid_matches)
        return winner, round(avg_score, 4), len(valid_matches), round(avg_quality, 1)
    
    # Case B: Winner-Take-All Fallback (Promotion of the best of the best)
    if tier_tallies:
        # Sort tallies by score and check the absolute highest one found across harvested frames
        best_tally = sorted(tier_tallies, key=lambda x: x['score'], reverse=True)[0]
        if best_tally['score'] >= PROMOTION_THRESHOLD:
            # Check for frequency consistency (did this tier win the most frames?)
            counts = Counter([t['tier'] for t in tier_tallies])
            freq_winner, freq_count = counts.most_common(1)[0]
            if freq_winner == best_tally['tier'] and freq_count >= (HARVEST_COUNT // 3):
                return freq_winner, round(best_tally['score'], 4), freq_count, round(avg_quality, 1)

    return "low_conf", 0.0, 0, round(avg_quality, 1)

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    results = {'floor_id': f_id, 'start_frame': start_f}
    
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss['tier'] == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_score"] = 1.0
            results[f"R{r+1}_S{c}_harv"] = 1
        return results

    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    f_range = range(start_f, end_f + 1)
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key] = "empty"
                results[f"{key}_score"] = 0.0
                results[f"{key}_harv"] = 0
            else:
                tier, score, harv, qual = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key] = tier
                results[f"{key}_score"] = score
                results[f"{key}_harv"] = harv
                results[f"{key}_qual"] = qual
                
    return results

def run_tier_identification():
    if not os.path.exists(DNA_INVENTORY_CSV):
        print(f"Error: DNA Inventory not found.")
        return

    df_floors = pd.read_csv(BOUNDARIES_CSV)
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    if LIMIT_FLOORS:
        df_floors = df_floors.head(LIMIT_FLOORS)
        print(f"--- DIAGNOSTIC MODE: LIMIT {LIMIT_FLOORS} FLOORS ---")
    else:
        print(f"--- STEP 4.2: TIER IDENTIFICATION (PRODUCTION RUN) ---")

    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    res = load_ore_templates()
    
    inventory = []
    worker = partial(process_floor_tier, dna_map=df_dna, buffer_dir=buffer_dir, all_files=all_files, res=res)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            f_id = result['floor_id']
            low_conf = sum(1 for k, v in result.items() if k.startswith('R') and v == 'low_conf')
            print(f"  Floor {f_id:03d}: Processed. LowConf Slots: {low_conf}")

    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Final Ore Inventory saved to: {OUT_CSV}")

    print("Generating visual audit proofs...")
    for _, row in final_df.iterrows():
        f_id = int(row['floor_id'])
        if not LIMIT_FLOORS and (f_id % 10 != 0 and f_id not in [1, 5, 25, 50, 75, 99]):
            continue
            
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier = row[key]
                if tier == "empty": continue
                score, harv, qual = row.get(f"{key}_score", 0), row.get(f"{key}_harv", 0), row.get(f"{key}_qual", 0)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                color = (0, 255, 0) if tier != "low_conf" else (0, 0, 255)
                hx, hy = cx + HUD_DX, cy + HUD_DY
                cv2.putText(img, f"{tier}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # Show Score (S), Harvester Count (H), and Quality (Q)
                cv2.putText(img, f"S:{score:.2f} H:{int(harv)} Q:{int(qual)}", (hx-25, hy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        cv2.imwrite(os.path.join(VERIFY_DIR, f"tier_audit_f{f_id:03d}.jpg"), img)

if __name__ == "__main__":
    run_tier_identification()