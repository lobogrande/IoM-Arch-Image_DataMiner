import sys, os
import cv2
import numpy as np
import pandas as pd

# Add root to sys.path to find project_config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIGURATION ---
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# Verified Template Filenames
SPRITE_R_NAME = "player_right.png"
SPRITE_L_NAME = "player_left.png"

# RECALIBRATED GRID CONSTANTS (Confirmed by Diagnostic)
GRID_X_START = 74 
GRID_Y_START = 261
STEP_X = 107.5
STEP_Y = 59.1
WAIT_OFFSET_X = 55   

# Detection Sensitivity
MATCH_THRESHOLD = 0.88 # Safe threshold based on 0.94 diagnostic result

def get_slot_center(slot_id):
    row = slot_id // 6
    col = slot_id % 6
    x = int(GRID_X_START + (col * STEP_X))
    y = int(GRID_Y_START + (row * STEP_Y))
    return x, y

def run_sprite_sequencer():
    print(f"--- SPRITE HOMING SEQUENCER (PRODUCTION): RUN {BUFFER_ID} ---")
    
    # 1. Load Templates & Get Dimensions
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, SPRITE_R_NAME), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, SPRITE_L_NAME), 0)
    
    if tpl_r is None or tpl_l is None:
        print(f"Error: Missing templates in {cfg.TEMPLATE_DIR}")
        return

    h_r, w_r = tpl_r.shape
    h_l, w_l = tpl_l.shape

    # 2. Setup Safe Homing Targets
    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    if not files: return
    
    # Get frame size for clamping
    sample_img = cv2.imread(os.path.join(SOURCE_DIR, files[0]), 0)
    img_h, img_w = sample_img.shape

    homing_targets = []
    # Row 1 (Slots 0-5) - Sprite on the Left
    for s_id in range(6):
        cx, cy = get_slot_center(s_id)
        tw, th = w_r + 20, h_r + 20 # 20px padding for stability
        tx = int(np.clip(cx - WAIT_OFFSET_X - (tw // 2), 0, img_w - tw))
        ty = int(np.clip(cy - (th // 2), 0, img_h - th))
        homing_targets.append({'slot': s_id, 'roi': (tx, ty, tw, th), 'template': tpl_r, 'facing': 'right'})
        
    # Slot 11 (Row 2) - Sprite on the Right (Floor 13 Transition)
    cx11, cy11 = get_slot_center(11)
    tw11, th11 = w_l + 20, h_l + 20
    tx11 = int(np.clip(cx11 + WAIT_OFFSET_X - (tw11 // 2), 0, img_w - tw11))
    ty11 = int(np.clip(cy11 - (th11 // 2), 0, img_h - th11))
    homing_targets.append({'slot': 11, 'roi': (tx11, ty11, tw11, th11), 'template': tpl_l, 'facing': 'left'})

    # 3. Execution Loop
    results = []
    print(f"Scanning {len(files)} frames...")

    for f_idx, filename in enumerate(files):
        img_path = os.path.join(SOURCE_DIR, filename)
        img = cv2.imread(img_path, 0)
        if img is None: continue

        for target in homing_targets:
            tx, ty, tw, th = target['roi']
            roi = img[ty:ty+th, tx:tx+tw]
            
            res = cv2.matchTemplate(roi, target['template'], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val >= MATCH_THRESHOLD:
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': target['slot'],
                    'facing': target['facing'],
                    'confidence': round(max_val, 4)
                })
                break 

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames analyzed...")

    # 4. Save and Verify
    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    
    if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 100:
        print(f"\n[SUCCESS] Cataloged {len(df)} sprite-adjacent frames.")
        print(f"Results saved to: {OUT_CSV}")
    else:
        print("\n[ERROR] CSV is empty or failed to save correctly.")

if __name__ == "__main__":
    run_sprite_sequencer()