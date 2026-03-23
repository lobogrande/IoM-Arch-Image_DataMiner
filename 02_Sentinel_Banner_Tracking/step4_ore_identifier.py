# step4_ore_identifier.py
# Purpose: Master Plan Step 4 - Identify all 24 ores on every floor using 
#          Parallelized Clean-Frame Search and Boss-Data Enforcement.
# Version: 1.3 (The Forensic Parallel Scanner)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

# GRID CONSTANTS (AI SENSOR CENTERS)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
AI_DIM = 48
SCALE = 1.20
SIDE_PX = int(AI_DIM * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# SCANNING LIMITS
MAX_SCAN_WINDOW = 100 # Maximum frames to search per floor for clean views
D_GATE = 6.5          # Occupancy threshold (diff vs BG)
MIN_ID_GATE = 0.55    # Minimum confidence to accept a plain match

def load_resources():
    """Pre-loads templates and masks into memory once."""
    res = {'ores': {}, 'bg': [], 'mask': None}
    t_path = cfg.TEMPLATE_DIR
    
    # Standard Circular Mask
    mask = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(mask, (SIDE_PX//2, SIDE_PX//2), int(18 * SCALE), 255, -1)
    res['mask'] = mask
    
    for f in os.listdir(t_path):
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        img = cv2.resize(img_raw, (SIDE_PX, SIDE_PX))
        
        if f.startswith("background"):
            res['bg'].append(img)
            continue
            
        if "_plain_" in f and "_act_" in f and not any(x in f for x in ["player", "negative"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = []
            res['ores'][tier].append(img)
            
    return res

def is_clean_active(roi_bgr, roi_gray):
    """Detects if a frame is usable (Active state and No Crosshair)."""
    # 1. Crosshair Detection (HSV Vibrancy)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    vibrant_mask = cv2.inRange(hsv, (0, 120, 120), (180, 255, 255))
    if cv2.countNonZero(vibrant_mask) > 100:
        return False
        
    # 2. Active State Detection (Laplacian Complexity)
    complexity = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
    if complexity < 300: # Typical shadow complexity
        return False
        
    return True

def identify_slot(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Forensic scan for a single slot across floor frames."""
    best_score = -1
    best_tier = "low_conf"
    
    # Coordinates
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2

    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Crop
        roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        roi_gray = img_gray[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        
        # Priority: Only scan frames that are Active and No-Crosshair
        if not is_clean_active(roi_bgr, roi_gray):
            continue
            
        for tier in allowed_tiers:
            if tier not in res['ores']: continue
            for tpl in res['ores'][tier]:
                res_match = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED, mask=res['mask'])
                score = cv2.minMaxLoc(res_match)[1]
                if score > best_score:
                    best_score = score
                    best_tier = tier
                    
        # If we found a high-confidence match in a clean frame, we can stop early
        if best_score > 0.85: break
        
    return best_tier, round(float(best_score), 4)

def process_floor(floor_data, buffer_dir, all_files, res):
    """Processes a single floor (Worker Function)."""
    f_id = int(floor_data['floor_id'])
    start_f, end_f = int(floor_data['true_start_frame']), int(floor_data['end_frame'])
    r4_dna, r3_dna = floor_data['dna_sig'].split('-')
    
    results = {'floor_id': f_id, 'start_frame': start_f}
    
    # 1. BOSS ENFORCEMENT
    if f_id in cfg.BOSS_DATA:
        boss = cfg.BOSS_DATA[f_id]
        for slot_idx in range(24):
            r, c = divmod(slot_idx, 6)
            identity = boss['special'][slot_idx] if boss['tier'] == 'mixed' else boss['tier']
            results[f"R{r+1}_S{c}"] = identity
            results[f"R{r+1}_S{c}_score"] = 1.0
        return results

    # 2. REGULAR FLOOR SCAN
    allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]
    f_range = range(start_f, min(end_f, start_f + MAX_SCAN_WINDOW))
    
    # Sample first frame for R1/R2 occupancy check
    img_sample = cv2.imread(os.path.join(buffer_dir, all_files[start_f]), 0)

    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            
            # Occupancy check
            occupied = False
            if r_idx == 2: occupied = r3_dna[col] == '1'
            elif r_idx == 3: occupied = r4_dna[col] == '1'
            else:
                # Top rows: manual subtraction check
                cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                roi = img_sample[cy-SIDE_PX//2:cy+SIDE_PX//2, cx-SIDE_PX//2:cx+SIDE_PX//2]
                if roi.shape == (SIDE_PX, SIDE_PX):
                    diff = min([np.sum(cv2.absdiff(roi, bg)) / (SIDE_PX**2) for bg in res['bg']])
                    occupied = diff > D_GATE
            
            if not occupied:
                results[key], results[f"{key}_score"] = "empty", 0.0
                continue
            
            # Forensic identification
            tier, score = identify_slot(f_range, r_idx, col, buffer_dir, all_files, allowed, res)
            results[key] = tier if score > MIN_ID_GATE else "low_conf"
            results[f"{key}_score"] = score
            
    return results

def run_ore_identification():
    if not os.path.exists(BOUNDARIES_CSV): return
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- STEP 4: ORE IDENTIFICATION v1.3 (Parallel Forensic) ---")
    
    # 1. Resource Pre-load
    res = load_resources()
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    # 2. Parallel Processing
    inventory = []
    print(f"Starting parallel scan of {len(df_floors)} floors...")
    
    worker = partial(process_floor, buffer_dir=buffer_dir, all_files=all_files, res=res)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row for _, row in df_floors.iterrows()}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            inventory.append(result)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(df_floors)} floors...", end="\r")

    # 3. Finalization & Reporting
    inventory_df = pd.DataFrame(inventory).sort_values('floor_id').reset_index(drop=True)
    inventory_df.to_csv(OUT_CSV, index=False)
    
    print(f"\n\n--- IDENTIFICATION COMPLETE ---")
    print(f"Inventory saved to: {OUT_CSV}")
    
    # Generate proof images for auditing
    print("Generating visual audit proofs (Start Frames)...")
    for idx, row in inventory_df.iterrows():
        if row['floor_id'] % 5 != 0: continue
        
        img_audit = cv2.imread(os.path.join(buffer_dir, all_files[int(row['start_frame'])]))
        if img_audit is None: continue
        
        for r_idx in range(4):
            for col in range(6):
                key = f"R{r_idx+1}_S{col}"
                identity = row[key]
                if identity == "empty": continue
                
                score = row[f"{key}_score"]
                ay = int(ORE0_Y + (r_idx * STEP))
                ax = int(ORE0_X + (col * STEP))
                color = (0, 255, 0) if identity != "low_conf" else (0, 0, 255)
                
                # Annotate
                cv2.putText(img_audit, f"{identity}", (ax+HUD_DX-25, ay+HUD_DY), 0, 0.4, (0,0,0), 2)
                cv2.putText(img_audit, f"{identity}", (ax+HUD_DX-25, ay+HUD_DY), 0, 0.4, color, 1)
                cv2.putText(img_audit, f"{score:.2f}", (ax+HUD_DX-25, ay+HUD_DY+12), 0, 0.3, (255,255,255), 1)

        cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{int(row['floor_id']):03d}.jpg"), img_audit)

if __name__ == "__main__":
    run_ore_identification()