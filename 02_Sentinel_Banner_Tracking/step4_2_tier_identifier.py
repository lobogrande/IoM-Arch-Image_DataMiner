# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers for occupied slots using 
#          Temporal Consensus and DNA Ingestion.
# Version: 1.1 (Telemetry & Diagnostic Enhancement)

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
LIMIT_FLOORS = 10  # Set to None for production; use small number for testing

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS
MIN_ID_CONFIDENCE = 0.50   # Minimum score to count a match toward consensus
MAX_HARVEST_FRAMES = 25    # Max frames per slot to harvest for consensus
COMPLEXITY_GATE = 300      # Laplacian floor for "Active" status

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
    """Detects if an ore is clean enough for tier identification."""
    # 1. Crosshair check (HSV)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Detect high saturation colors typical of targeting UI
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    if cv2.countNonZero(mask) > 130: return False
    
    # 2. Active state check (Complexity)
    if cv2.Laplacian(roi_gray, cv2.CV_64F).var() < COMPLEXITY_GATE: return False
    
    return True

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Scans the floor to find a consensus identity via temporal voting."""
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
            
            if best_t and best_s >= MIN_ID_CONFIDENCE:
                harvested_matches.append(best_t)
                
        if len(harvested_matches) >= MAX_HARVEST_FRAMES: break

    if not harvested_matches:
        return "low_conf", 0.0, 0
    
    winner, win_count = Counter(harvested_matches).most_common(1)[0]
    conf = win_count / len(harvested_matches)
    return winner, round(conf, 2), len(harvested_matches)

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
            results[f"R{r+1}_S{c}_conf"] = 1.0
            results[f"R{r+1}_S{c}_harv"] = 1
        return results

    # 2. TIER RESTRICTION FILTER
    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    f_range = range(start_f, end_f + 1)
    
    # 3. IDENTIFY OCCUPIED SLOTS
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    total_occupied = 0
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0':
                results[key] = "empty"
                results[f"{key}_conf"] = 0.0
                results[f"{key}_harv"] = 0
            else:
                total_occupied += 1
                tier, conf, harv = identify_consensus(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
                results[key] = tier
                results[f"{key}_conf"] = conf
                results[f"{key}_harv"] = harv
                
    results['occupied_count'] = total_occupied
    return results

def run_tier_identification():
    if not os.path.exists(DNA_INVENTORY_CSV):
        print(f"Error: DNA Inventory not found. Run Step 4.1 first.")
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
            
            # Real-time console logging
            f_id = result['floor_id']
            occ = result.get('occupied_count', 0)
            low_conf = sum(1 for k, v in result.items() if k.startswith('R') and v == 'low_conf')
            print(f"  Floor {f_id:03d}: Processed. Occupied: {occ}, LowConf: {low_conf}")

    # Save Results
    final_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    final_df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Final Ore Inventory saved to: {OUT_CSV}")

    # Generate Audit Proofs
    print("Generating visual audit proofs...")
    for _, row in final_df.iterrows():
        f_id = int(row['floor_id'])
        # In diagnostic mode, we save EVERY floor. In production, we sample.
        if not LIMIT_FLOORS and (f_id % 10 != 0 and f_id not in [1, 5, 25, 50, 75, 99]):
            continue
            
        img = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img is None: continue
        
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                tier = row[key]
                if tier == "empty": continue
                
                conf = row[f"{key}_conf"]
                harv = row[f"{key}_harv"]
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                color = (0, 255, 0) if tier != "low_conf" else (0, 0, 255)
                
                hx, hy = cx + HUD_DX, cy + HUD_DY
                cv2.putText(img, f"{tier}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
                cv2.putText(img, f"{tier}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                # Show harvester count for diagnostics
                cv2.putText(img, f"H:{harv} C:{conf:.1f}", (hx-25, hy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

        cv2.imwrite(os.path.join(VERIFY_DIR, f"tier_audit_f{f_id:03d}.jpg"), img)

if __name__ == "__main__":
    run_tier_identification()