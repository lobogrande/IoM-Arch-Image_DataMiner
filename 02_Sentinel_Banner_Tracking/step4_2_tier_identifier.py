# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers for occupied slots using 
#          Adaptive Temporal Harvesting and DNA Ingestion.
# Version: 1.2 (The Adaptive Grader)

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
LIMIT_FLOORS = 20  # Set to None for production; use small number for testing

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS
MIN_MATCH_CONFIDENCE = 0.45  # Lowered slightly for early-game ores
HARVEST_COUNT = 15          # Number of best-quality frames to use for consensus

def load_ore_templates():
    """Pre-loads pristine active templates (48x48) into memory."""
    res = {'ores': {}}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        # We only use _plain_ and _act_ (Active) templates for identification
        if "_plain_" in f and "_act_" in f and not any(x in f for x in ["player", "negative", "background"]):
            img_raw = cv2.imread(os.path.join(t_path, f), 0)
            if img_raw is not None:
                tier = f.split("_")[0]
                if tier not in res['ores']: res['ores'][tier] = []
                # Keep templates at native 48x48 for the sliding ROI match
                res['ores'][tier].append(cv2.resize(img_raw, (48, 48)))
    return res

def get_frame_quality(roi_bgr, roi_gray):
    """Calculates a quality score for a frame (Higher is cleaner)."""
    # 1. UI Noise Penalty (HSV)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    ui_pixels = cv2.countNonZero(mask)
    ui_penalty = ui_pixels * 0.5
    
    # 2. Activity Reward (Complexity)
    complexity = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    
    # Quality = Structural Signal minus UI Noise
    return complexity - ui_penalty

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Grades all frames, picks the best ones, and votes on identity."""
    frame_candidates = []
    
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 # 57px target

    # Phase 1: Quality Grading
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        
        roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        
        quality = get_frame_quality(roi_bgr, roi_gray)
        # Store index and quality for ranking
        frame_candidates.append({'idx': f_idx, 'quality': quality, 'gray': roi_gray})

    if not frame_candidates:
        return "low_conf", 0.0, 0, 0.0

    # Phase 2: Harvest the Top N Frames
    top_frames = sorted(frame_candidates, key=lambda x: x['quality'], reverse=True)[:HARVEST_COUNT]
    
    # Phase 3: Identify Tiers in Harvested Frames
    tier_matches = []
    total_score = 0.0
    
    # 30x30 central ROI from our 57x57 crop
    # This matches the Step 4.1 DNA profile logic
    roi_side = 30
    offset = (SIDE_PX - roi_side) // 2

    for frame in top_frames:
        roi_30 = frame['gray'][offset:offset+roi_side, offset:offset+roi_side]
        
        best_f_score = -1
        best_f_tier = None
        
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            for tpl in res['ores'][tier]:
                # ROI (30x30) slides across Template (48x48)
                match_res = cv2.matchTemplate(tpl, roi_30, cv2.TM_CCOEFF_NORMED)
                score = cv2.minMaxLoc(match_res)[1]
                if score > best_f_score:
                    best_f_score = score
                    best_f_tier = tier
        
        if best_f_tier and best_f_score >= MIN_MATCH_CONFIDENCE:
            tier_matches.append(best_f_tier)
            total_score += best_f_score

    if not tier_matches:
        avg_q = np.mean([f['quality'] for f in top_frames])
        return "low_conf", 0.0, 0, avg_q
    
    # Consensus results
    counts = Counter(tier_matches)
    winner, win_count = counts.most_common(1)[0]
    avg_score = total_score / len(tier_matches)
    avg_quality = np.mean([f['quality'] for f in top_frames])
    
    return winner, round(avg_score, 4), len(tier_matches), round(avg_quality, 1)

def process_floor_tier(floor_data, dna_map, buffer_dir, all_files, res):
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    results = {'floor_id': f_id, 'start_frame': start_f}
    
    # 1. BOSS DATA ENFORCEMENT
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for s_idx in range(24):
            r, c = divmod(s_idx, 6)
            identity = boss['special'][s_idx] if boss['tier'] == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_score"] = 1.0
            results[f"R{r+1}_S{c}_harv"] = 1
        return results

    # 2. TIER RESTRICTION FILTER
    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    f_range = range(start_f, end_f + 1)
    
    # 3. IDENTIFY OCCUPIED SLOTS
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            # Step 4.1 ground truth check
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
    
    print(f"Parallelizing identification of {len(df_floors)} floors...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            
            f_id = result['floor_id']
            low_conf = sum(1 for k, v in result.items() if k.startswith('R') and v == 'low_conf')
            print(f"  Floor {f_id:03d}: Processed. LowConf Slots: {low_conf}")

    # Save Results
    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Final Ore Inventory saved to: {OUT_CSV}")

    # Generate Audit Proofs
    print("Generating visual audit proofs...")
    for _, row in final_df.iterrows():
        f_id = int(row['floor_id'])
        # Show all in diagnostic mode; sample in production
        if not LIMIT_FLOORS and (f_id % 10 != 0 and f_id not in [1, 5, 25, 50, 75, 99]):
            continue
            
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier = row[key]
                if tier == "empty": continue
                
                score = row.get(f"{key}_score", 0)
                harv = row.get(f"{key}_harv", 0)
                qual = row.get(f"{key}_qual", 0)
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                
                color = (0, 255, 0) if tier != "low_conf" else (0, 0, 255)
                hx, hy = cx + HUD_DX, cy + HUD_DY
                cv2.putText(img, f"{tier}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # Show forensic metrics
                cv2.putText(img, f"S:{score:.2f} H:{int(harv)}", (hx-25, hy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        cv2.imwrite(os.path.join(VERIFY_DIR, f"tier_audit_f{f_id:03d}.jpg"), img)

if __name__ == "__main__":
    run_tier_identification()