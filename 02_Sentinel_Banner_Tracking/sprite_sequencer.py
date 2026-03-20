# sprite_sequencer.py
# Version: 1.4
# Refactor: Multi-row full-width scan for 100% Slot 0-5 and Slot 11 coverage

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.4"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Grid Geometry Constants
GRID_X_START = 74
GRID_Y_START = 261
STEP_X = 107.5
STEP_Y = 59.1
SPRITE_OFFSET_X = 55 # Distance from sprite to slot center

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Full-Grid Mode) ---")
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        # --- SCAN 1: ROW 1 (Slots 0-5) ---
        # Search Strip covers full grid width
        r1_y = GRID_Y_START
        strip1 = img[r1_y-40 : r1_y+40, 0:1000] 
        res_r = cv2.matchTemplate(strip1, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_r)

        if max_val_r > 0.85:
            found_x = max_loc_r[0] + SPRITE_OFFSET_X
            inferred_slot = round((found_x - GRID_X_START) / STEP_X)
            if 0 <= inferred_slot <= 5:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': inferred_slot, 'confidence': round(max_val_r, 4)})

        # --- SCAN 2: ROW 2 (Specifically Slot 11) ---
        # Search Strip for the second row transition
        r2_y = int(GRID_Y_START + STEP_Y)
        strip2 = img[r2_y-40 : r2_y+40, 800:1500] # Focus on the right side for Slot 11
        res_l = cv2.matchTemplate(strip2, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_l)

        if max_val_l > 0.85:
            # For Slot 11, player is to the RIGHT, so we subtract offset
            found_x_l = (max_loc_l[0] + 800) - SPRITE_OFFSET_X 
            inferred_slot_l = round((found_x_l - GRID_X_START) / STEP_X)
            if inferred_slot_l == 11:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': 11, 'confidence': round(max_val_l, 4)})

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames... (Detections: {len(results)})")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} saved {len(results)} detections to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()