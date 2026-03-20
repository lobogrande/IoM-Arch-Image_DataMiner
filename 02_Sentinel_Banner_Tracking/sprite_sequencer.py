# sprite_sequencer.py
# Version: 1.5
# Fix: Dynamic strip sizing and boundary clamping to prevent OpenCV assertion errors.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.5"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Grid Geometry Constants
GRID_X_START = 74
GRID_Y_START = 261
STEP_X = 107.5
STEP_Y = 59.1
SPRITE_OFFSET_X = 55 

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Dynamic-Scan Mode) ---")
    
    # 1. Load Templates and Determine Safe Search Height
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    if tpl_r is None or tpl_l is None:
        print("Error: Missing player_right.png or player_left.png in Reference folder.")
        return

    # Measure templates to ensure our search strips are always large enough
    h_r, w_r = tpl_r.shape
    h_l, w_l = tpl_l.shape
    strip_h = max(h_r, h_l) + 20  # Template height + 20px padding
    half_h = strip_h // 2

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    # Get frame dimensions for safe clamping
    sample_img = cv2.imread(os.path.join(SOURCE_DIR, files[0]), 0)
    img_h, img_w = sample_img.shape

    # Define X-Scan ranges
    R1_X_START, R1_X_END = 0, min(1000, img_w)
    R2_X_START, R2_X_END = 800, min(1600, img_w)

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        # --- SCAN 1: ROW 1 (Slots 0-5) ---
        r1_y = GRID_Y_START
        y1, y2 = int(np.clip(r1_y - half_h, 0, img_h - strip_h)), int(np.clip(r1_y + half_h, strip_h, img_h))
        strip1 = img[y1:y2, R1_X_START : R1_X_END]
        
        res_r = cv2.matchTemplate(strip1, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val_r, _, max_loc_r = cv2.minMaxLoc(res_r)

        if max_val_r > 0.85:
            found_x = max_loc_r[0] + SPRITE_OFFSET_X
            inferred_slot = round((found_x - GRID_X_START) / STEP_X)
            if 0 <= inferred_slot <= 5:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': inferred_slot, 'confidence': round(max_val_r, 4)})

        # --- SCAN 2: ROW 2 (Slot 11) ---
        r2_y = int(GRID_Y_START + STEP_Y)
        y3, y4 = int(np.clip(r2_y - half_h, 0, img_h - strip_h)), int(np.clip(r2_y + half_h, strip_h, img_h))
        strip2 = img[y3:y4, R2_X_START : R2_X_END]
        
        res_l = cv2.matchTemplate(strip2, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, max_val_l, _, max_loc_l = cv2.minMaxLoc(res_l)

        if max_val_l > 0.85:
            found_x_l = (max_loc_l[0] + R2_X_START) - SPRITE_OFFSET_X 
            inferred_slot_l = round((found_x_l - GRID_X_START) / STEP_X)
            if inferred_slot_l == 11:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': 11, 'confidence': round(max_val_l, 4)})

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames... (Detections: {len(results)})")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} saved {len(results)} detections to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()