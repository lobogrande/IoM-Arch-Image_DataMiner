import sys, os
import cv2
import numpy as np
import pandas as pd

# Find the project root from a sub-folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIGURATION ---
BUFFER_ID = 0  # Start with Run_0
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Sprite Templates (Assumed names in Data_01_Reference/templates)
SPRITE_R = os.path.join(cfg.TEMPLATE_DIR, "player_facing_right.png")
SPRITE_L = os.path.join(cfg.TEMPLATE_DIR, "player_facing_left.png")

# Grid Constants (Recalibrated from 3-stage worker scripts)
GRID_X_START = 64
GRID_Y_START = 261
STEP_X = 107.5
STEP_Y = 59.1

# Sprite Homing Offsets (Adjusted to sit next to the slot)
WAIT_OFFSET_X = 50  # Pixels to the left/right of slot center
SEARCH_WINDOW = 40  # Size of the ROI to search for sprite

def get_slot_coords(slot_id):
    row = slot_id // 6
    col = slot_id % 6
    x = int(GRID_X_START + (col * STEP_X))
    y = int(GRID_Y_START + (row * STEP_Y))
    return x, y

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER ALPHA: RUN {BUFFER_ID} ---")
    
    # Load Templates
    tpl_r = cv2.imread(SPRITE_R, 0)
    tpl_l = cv2.imread(SPRITE_L, 0)
    
    if tpl_r is None or tpl_l is None:
        print("Error: Player sprite templates not found in Reference folder.")
        return

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    # Define our 7 Homing Targets (Slots 0-5 and Slot 11)
    homing_targets = []
    # Slots 0-5: Sprite on the Left
    for s_id in range(6):
        x, y = get_slot_coords(s_id)
        homing_targets.append({
            'slot': s_id, 
            'roi': (x - WAIT_OFFSET_X - 20, y - 20, SEARCH_WINDOW, SEARCH_WINDOW),
            'template': tpl_r
        })
    # Slot 11: Sprite on the Right
    x11, y11 = get_slot_coords(11)
    homing_targets.append({
        'slot': 11, 
        'roi': (x11 + WAIT_OFFSET_X - 20, y11 - 20, SEARCH_WINDOW, SEARCH_WINDOW),
        'template': tpl_l
    })

    print(f"Scanning {len(files)} frames for sprite activity...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        for target in homing_targets:
            tx, ty, tw, th = target['roi']
            roi = img[ty:ty+th, tx:tx+tw]
            
            # Template Match
            res = cv2.matchTemplate(roi, target['template'], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > 0.85:
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': target['slot'],
                    'confidence': round(max_val, 3)
                })
                # Break once sprite found in a target for this frame
                break

        if f_idx % 1000 == 0:
            print(f"  [Progress] Processed {f_idx} frames...")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"--- SCAN COMPLETE ---")
    print(f"Detected sprite in {len(df)} frames. Log saved to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()