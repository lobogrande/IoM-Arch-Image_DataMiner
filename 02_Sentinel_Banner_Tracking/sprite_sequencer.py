# sprite_sequencer.py
# Version: 1.8
# Refactor: Headless Template Matching & Full-Grid Spatial Discovery

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.8"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Calibrated Grid Logic
GRID_X_START, GRID_Y_START = 74, 261
STEP_X, STEP_Y = 107.5, 59.1
SPRITE_OFFSET_X = 55 

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Headless Global Scan) ---")
    
    # 1. Load and "Behead" Templates (Crop top 30%)
    raw_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    raw_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    if raw_r is None or raw_l is None: return

    # Crop off the top 35% to avoid 'Dig Stage' UI text
    h_r, w_r = raw_r.shape
    tpl_r = raw_r[int(h_r*0.35):, :]
    
    h_l, w_l = raw_l.shape
    tpl_l = raw_l[int(h_l*0.35):, :]

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    print(f"Scanning {len(files)} frames for headless sprite...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        # Define the Search Zone (The Grid Area)
        # Y: 200 to 600, X: 0 to 1200
        search_zone = img[200:600, 0:1200]
        
        # --- Search for Player Facing Right (Slots 0-5) ---
        res_r = cv2.matchTemplate(search_zone, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_r)

        if max_val_r > 0.82:
            # Calculate Slot based on X-Position (add 200 back for Y, and 0 for X)
            sprite_center_x = max_loc_r[0] + (w_r // 2)
            found_slot_x = sprite_center_x + SPRITE_OFFSET_X
            inferred_slot = round((found_slot_x - GRID_X_START) / STEP_X)
            
            if 0 <= inferred_slot <= 5:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': inferred_slot, 'confidence': round(max_val_r, 4)})
                continue # If found here, skip the left-facing search

        # --- Search for Player Facing Left (Slot 11) ---
        res_l = cv2.matchTemplate(search_zone, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_l)

        if max_val_l > 0.82:
            sprite_center_x_l = max_loc_l[0] + (w_l // 2)
            found_slot_x_l = sprite_center_x_l - SPRITE_OFFSET_X
            inferred_slot_l = round((found_slot_x_l - GRID_X_START) / STEP_X)
            
            if inferred_slot_l == 11:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': 11, 'confidence': round(max_val_l, 4)})

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames... (Detections: {len(results)})")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} saved {len(results)} detections.")

if __name__ == "__main__":
    run_sprite_sequencer()