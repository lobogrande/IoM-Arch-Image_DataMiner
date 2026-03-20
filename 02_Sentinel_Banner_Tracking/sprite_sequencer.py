# sprite_sequencer.py
# Version: 1.3
# Refactor: Full-width horizontal scan to fix missing slot detections

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.3"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Grid Geometry
GRID_Y_START = 261
STEP_X = 107.5
SCAN_HEIGHT = 80 # Vertical height of the search strip

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Full-Scan Mode) ---")
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        # SCAN ROW 1: We look across the entire width of the grid (x: 0 to 800)
        # centered on the Row 1 Y-coordinate
        row1_strip = img[GRID_Y_START-40 : GRID_Y_START+40, 0:850]
        
        # Match Facing Right
        res_r = cv2.matchTemplate(row1_strip, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_r)

        if max_val_r > 0.85:
            # Calculate which slot this X-coordinate corresponds to
            # max_loc_r[0] is the X pixel where the sprite was found
            found_x = max_loc_r[0] + 55 # Add back the offset to get to slot center
            inferred_slot = round((found_x - 74) / STEP_X)
            
            if 0 <= inferred_slot <= 5:
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': inferred_slot,
                    'x_pixel': max_loc_r[0],
                    'confidence': round(max_val_r, 4)
                })

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames... (Found: {len(results)})")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} saved {len(results)} detections to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()