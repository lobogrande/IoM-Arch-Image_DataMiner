# step4_ore_identifier.py
# Purpose: Master Plan Step 4 - Identify all ores on every floor using 
#          multi-frame pristine selection and DNA-masked scanning.
# Version: 1.0 (The Pristine Scanner)

import sys, os, cv2, numpy as np, pandas as pd
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

# HUD VERIFICATION OFFSETS (README.md)
HUD_DX, HUD_DY = 20, 30

# SEARCH SETTINGS
PRISTINE_WINDOW = 15 # Look at first 15 frames to find cleanest view
MATCH_THRESHOLD = 0.45

def load_plain_templates():
    """Loads only ore templates with the '_plain_' modifier."""
    templates = {}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path):
        return templates
    
    for f in os.listdir(t_path):
        # Filter for plain ore templates only
        if "_plain_" in f and not any(x in f for x in ["background", "player", "negative"]):
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None:
                # Resize to our consistent AI scale
                tier = f.split("_")[0]
                if tier not in templates: templates[tier] = []
                templates[tier].append(cv2.resize(img, (SIDE_PX, SIDE_PX)))
    return templates

def get_slot_roi(img_gray, row_idx, col_idx):
    """Extracts the 57x57 ROI centered on the slot."""
    y_center = int(ORE0_Y + (row_idx * STEP))
    x_center = int(ORE0_X + (col_idx * STEP))
    
    y1, x1 = y_center - (SIDE_PX // 2), x_center - (SIDE_PX // 2)
    roi = img_gray[y1 : y1 + SIDE_PX, x1 : x1 + SIDE_PX]
    
    if roi.shape != (SIDE_PX, SIDE_PX):
        return None
    return roi

def identify_ore(roi, templates):
    """Matches an ROI against the plain template library."""
    best_score = -1
    best_tier = "unknown"
    
    for tier, tpls in templates.items():
        for tpl in tpls:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_tier = tier
    
    return best_tier, round(float(best_score), 4)

def run_ore_identification():
    if not os.path.exists(BOUNDARIES_CSV):
        print(f"Error: {BOUNDARIES_CSV} not found.")
        return

    df_floors = pd.read_csv(BOUNDARIES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    ore_tpls = load_plain_templates()
    
    if not os.path.exists(VERIFY_DIR): os.makedirs(VERIFY_DIR)
    
    print(f"--- STEP 4: ORE IDENTIFICATION (110 Floors) ---")
    print(f"Using {len(ore_tpls)} template tiers for identification...")

    inventory = []

    for i, floor in df_floors.iterrows():
        floor_id = int(floor['floor_id'])
        start_f = int(floor['true_start_frame'])
        end_f = int(floor['end_frame'])
        
        # Parse DNA into row-occupancy masks
        r4_dna, r3_dna = floor['dna_sig'].split('-')
        
        # 1. FIND PRISTINE FRAME
        # We check the first few frames to see which one gives us the best match scores.
        # This bypasses frames where a banner might be scrolling through the grid.
        best_overall_score = -1
        pristine_idx = start_f
        
        search_limit = min(end_f, start_f + PRISTINE_WINDOW)
        for f_idx in range(start_f, search_limit + 1):
            img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
            if img is None: continue
            
            current_total_score = 0
            count = 0
            # Sample Row 4 (usually cleanest) to determine frame quality
            for col in range(6):
                if r4_dna[col] == '1':
                    roi = get_slot_roi(img, 3, col)
                    if roi is not None:
                        _, score = identify_ore(roi, ore_tpls)
                        current_total_score += score
                        count += 1
            
            avg_score = current_total_score / max(1, count)
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                pristine_idx = f_idx
        
        # 2. PERFORM FINAL IDENTIFICATION ON PRISTINE FRAME
        print(f"Floor {floor_id:03d}: Identifying on Frame {pristine_idx}...", end="\r")
        img_pristine = cv2.imread(os.path.join(buffer_dir, all_files[pristine_idx]))
        img_gray = cv2.cvtColor(img_pristine, cv2.COLOR_BGR2GRAY)
        
        floor_ores = {'floor_id': floor_id, 'pristine_frame': pristine_idx}
        
        # Process Row 3 (Slots 0-5) and Row 4 (Slots 6-11)
        # Note: We represent the grid as a flat 12-slot array
        for r_idx, dna_str in [(2, r3_dna), (3, r4_dna)]:
            row_label = "R3" if r_idx == 2 else "R4"
            for col in range(6):
                slot_key = f"{row_label}_S{col}"
                if dna_str[col] == '1':
                    roi = get_slot_roi(img_gray, r_idx, col)
                    tier, score = identify_ore(roi, ore_tpls)
                    
                    is_valid = score >= MATCH_THRESHOLD
                    identity = tier if is_valid else "low_conf"
                    floor_ores[slot_key] = identity
                    floor_ores[f"{slot_key}_score"] = score
                    
                    # Diagnostic Drawing
                    # AI Center
                    ax = int(ORE0_X + (col * STEP))
                    ay = int(ORE0_Y + (r_idx * STEP))
                    # HUD Drawing coords
                    hx, hy = ax + HUD_DX, ay + HUD_DY
                    
                    color = (0, 255, 0) if is_valid else (0, 0, 255)
                    cv2.putText(img_pristine, identity, (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    floor_ores[slot_key] = "empty"
                    floor_ores[f"{slot_key}_score"] = 0.0

        inventory.append(floor_ores)
        
        # Periodic visual export
        if floor_id % 5 == 0:
            out_name = f"ore_id_f{floor_id:03d}_frame_{pristine_idx}.jpg"
            cv2.imwrite(os.path.join(VERIFY_DIR, out_name), img_pristine)

    # 3. SAVE RESULTS
    out_df = pd.DataFrame(inventory)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Identified ores for {len(inventory)} floors.")
    print(f"Inventory saved to: {OUT_CSV}")
    print(f"Visual proofs saved to: {VERIFY_DIR}")

if __name__ == "__main__":
    run_ore_identification()