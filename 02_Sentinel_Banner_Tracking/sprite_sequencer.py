# sprite_sequencer.py
# Version: 1.7
# Fix: Implemented pre-call dimension validation to prevent OpenCV termination.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.7"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Grid Geometry Constants
GRID_X_START = 74
GRID_Y_START = 261
STEP_X = 107.5
STEP_Y = 59.1
SPRITE_OFFSET_X = 55 # Distance from sprite center to slot center

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Stabilized Mode) ---")
    
    # 1. Load Templates
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    if tpl_r is None or tpl_l is None:
        print("Error: Player templates not found. Check Reference folder.")
        return

    h_r, w_r = tpl_r.shape
    h_l, w_l = tpl_l.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    for f_idx, filename in enumerate(files):
        img_path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(img_path, 0)
        if img is None: continue
        ih, iw = img.shape

        # --- SCAN 1: ROW 1 (Slots 0-5) ---
        # Search Strip for player facing RIGHT (tpl_r)
        y1, y2 = max(0, GRID_Y_START - 50), min(ih, GRID_Y_START + 50)
        strip1 = img[y1:y2, 0:iw]
        
        # Explicit Safety Check
        if strip1.shape[0] >= h_r and strip1.shape[1] >= w_r:
            res_r = cv2.matchTemplate(strip1, tpl_r, cv2.TM_CCOEFF_NORMED)
            _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_r)
            
            if max_val_r > 0.85:
                # Logic: Find sprite center, then apply wait offset to find Slot center
                found_x = max_loc_r[0] + (w_r // 2) + SPRITE_OFFSET_X
                inferred_slot = round((found_x - GRID_X_START) / STEP_X)
                
                if 0 <= inferred_slot <= 5:
                    results.append({
                        'frame_idx': f_idx, 
                        'filename': filename, 
                        'slot_id': inferred_slot, 
                        'confidence': round(max_val_r, 4)
                    })

        # --- SCAN 2: ROW 2 (Specifically Slot 11) ---
        # Search Strip for player facing LEFT (tpl_l)
        r2_y = int(GRID_Y_START + STEP_Y)
        y3, y4 = max(0, r2_y - 50), min(ih, r2_y + 50)
        strip2 = img[y3:y4, 0:iw]
        
        if strip2.shape[0] >= h_l and strip2.shape[1] >= w_l:
            res_l = cv2.matchTemplate(strip2, tpl_l, cv2.TM_CCOEFF_NORMED)
            _, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_l)
            
            if max_val_l > 0.85:
                # Logic: Player sits to the RIGHT of Slot 11 center
                found_x_l = max_loc_l[0] + (w_l // 2) - SPRITE_OFFSET_X
                inferred_slot_l = round((found_x_l - GRID_X_START) / STEP_X)
                
                if inferred_slot_l == 11:
                    results.append({
                        'frame_idx': f_idx, 
                        'filename': filename, 
                        'slot_id': 11, 
                        'confidence': round(max_val_l, 4)
                    })

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames... (Hits: {len(results)})")

    # Final Save
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} saved {len(results)} detections to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()