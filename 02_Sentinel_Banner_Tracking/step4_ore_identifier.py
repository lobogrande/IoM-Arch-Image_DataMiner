# step4_ore_identifier.py
# Purpose: Master Plan Step 4 - Identify all ores on every floor using 
#          independent per-slot scanning and start-frame anchoring.
# Version: 1.1 (The Independent Slot Scalpel)

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
PRISTINE_WINDOW = 15 # Look at first 15 frames to find cleanest view for EACH slot
MATCH_THRESHOLD = 0.55 # Raised threshold for "Pristine" plain templates

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
    print(f"Using {len(ore_tpls)} template tiers for per-slot independent scanning...")

    inventory = []

    for i, floor in df_floors.iterrows():
        floor_id = int(floor['floor_id'])
        start_f = int(floor['true_start_frame'])
        end_f = int(floor['end_frame'])
        
        # Parse DNA into row-occupancy masks
        r4_dna, r3_dna = floor['dna_sig'].split('-')
        
        # 1. PER-SLOT INDEPENDENT SCANNING
        # Instead of one "pristine" frame, we find the best frame for every single slot.
        # This bypasses different ores being blocked by different noise at different times.
        floor_results = {'floor_id': floor_id, 'start_frame': start_f}
        search_limit = min(end_f, start_f + PRISTINE_WINDOW)
        
        print(f"Floor {floor_id:03d}: Forensic Scanning {start_f} -> {search_limit}...", end="\r")

        for r_idx, dna_str in [(2, r3_dna), (3, r4_dna)]:
            row_label = "R3" if r_idx == 2 else "R4"
            for col in range(6):
                slot_key = f"{row_label}_S{col}"
                
                if dna_str[col] == '1':
                    best_slot_score = -1
                    best_slot_tier = "low_conf"
                    
                    # Scan the window specifically for THIS slot
                    for f_idx in range(start_f, search_limit + 1):
                        img_gray = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
                        if img_gray is None: continue
                        
                        roi = get_slot_roi(img_gray, r_idx, col)
                        if roi is not None:
                            tier, score = identify_ore(roi, ore_tpls)
                            if score > best_slot_score:
                                best_slot_score = score
                                best_slot_tier = tier
                    
                    # Lock in the best identity found across the window
                    is_valid = best_slot_score >= MATCH_THRESHOLD
                    floor_results[slot_key] = best_slot_tier if is_valid else "low_conf"
                    floor_results[f"{slot_key}_score"] = best_slot_score
                else:
                    floor_results[slot_key] = "empty"
                    floor_results[f"{slot_key}_score"] = 0.0

        inventory.append(floor_results)
        
        # 2. VISUAL PROOF GENERATION (Always Anchored to Start Frame)
        # Regardless of which frame the ore was found in, we draw on the start frame for audit.
        img_anchor = cv2.imread(os.path.join(buffer_dir, all_files[start_f]))
        if img_anchor is not None:
            for r_idx, row_label in [(2, "R3"), (3, "R4")]:
                for col in range(6):
                    slot_key = f"{row_label}_S{col}"
                    if floor_results[slot_key] == "empty": continue
                    
                    identity = floor_results[slot_key]
                    score = floor_results[f"{slot_key}_score"]
                    is_valid = identity != "low_conf"
                    
                    # Coordinates
                    ax = int(ORE0_X + (col * STEP))
                    ay = int(ORE0_Y + (r_idx * STEP))
                    hx, hy = ax + HUD_DX, ay + HUD_DY
                    
                    color = (0, 255, 0) if is_valid else (0, 0, 255)
                    # Text annotation with shadow for legibility
                    cv2.putText(img_anchor, f"{identity}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
                    cv2.putText(img_anchor, f"{identity}", (hx-25, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.putText(img_anchor, f"{score:.2f}", (hx-25, hy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)

            out_name = f"floor_{floor_id:03d}_start_frame_{start_f}.jpg"
            cv2.imwrite(os.path.join(VERIFY_DIR, out_name), img_anchor)

    # 3. SAVE RESULTS & SUCCESS DIAGNOSTIC
    out_df = pd.DataFrame(inventory)
    out_df.to_csv(OUT_CSV, index=False)
    
    # Audit logic: How many "low_conf" per floor?
    low_conf_cols = [c for c in out_df.columns if not c.endswith("_score") and c not in ["floor_id", "start_frame"]]
    out_df['low_conf_count'] = (out_df[low_conf_cols] == "low_conf").sum(axis=1)
    
    print(f"\n--- IDENTIFICATION SUMMARY ---")
    print(f"Total Ores Identified: {(out_df[low_conf_cols] != 'low_conf').sum().sum() - (out_df[low_conf_cols] == 'empty').sum().sum()}")
    print(f"Low Confidence Detections: {(out_df[low_conf_cols] == 'low_conf').sum().sum()}")
    print(f"Floors requiring audit (3+ low_conf): {len(out_df[out_df['low_conf_count'] >= 3])}")
    
    print(f"\nInventory saved to: {OUT_CSV}")
    print(f"Visual proofs (Anchored to Start Frames) saved to: {VERIFY_DIR}")

if __name__ == "__main__":
    run_ore_identification()