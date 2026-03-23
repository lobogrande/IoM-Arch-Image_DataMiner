# step1_sprite_homing.py
# Purpose: Step 1 - Spatial detection of mining events using centralized config.
# Version: 3.0 (Architecture Aligned)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIGURATION ---
# Pulls path directly from config default (e.g., capture_buffer_1)
SOURCE_DIR = cfg.get_buffer_path() 

# Determine Run ID from the folder name for output naming
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{RUN_ID}.csv")

# --- VALIDATED CONSTANTS (Lava Biome) ---
ORE0_HUD_X, ORE0_HUD_Y = 74, 261
STEP = 59.0
PLAYER_OFFSET = 41.0
TPL_W, TPL_H = 40, 60

# Staircase Thresholds
STAIRCASE = {0: 0.90, 1: 0.85, 2: 0.82, 3: 0.78, 4: 0.75, 5: 0.72, 11: 0.82}

def get_slot_geometry(slot_id):
    col = slot_id % 6
    row = slot_id // 6
    hox = int(ORE0_HUD_X + (col * STEP))
    hoy = int(ORE0_HUD_Y + (row * STEP))
    hpx = int(hox - PLAYER_OFFSET) if slot_id < 6 else int(hox + PLAYER_OFFSET)
    apx, apy = int(hpx - (TPL_W // 2)), int(hoy - (TPL_H // 2))
    return (apx, apy)

def run_homing_scan():
    print(f"--- STEP 1: SPRITE HOMING (Target: {os.path.basename(SOURCE_DIR)}) ---")
    full_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    full_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if full_r is None or full_l is None:
        print("Error: Player templates missing.")
        return
    
    bot_r, bot_l = full_r[30:, :], full_l[30:, :]
    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    
    targets = []
    for s_id in sorted(STAIRCASE.keys()):
        ax, ay = get_slot_geometry(s_id)
        targets.append({
            'id': s_id, 'x': ax, 'y': ay,
            'tpl_f': full_r if s_id < 6 else full_l,
            'tpl_b': bot_r if s_id < 6 else bot_l,
            'thresh': STAIRCASE[s_id]
        })

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        for t in targets:
            if t['x'] < 0 or t['y'] < 0: continue
            roi_f = img[t['y']:t['y']+60, t['x']:t['x']+40]
            if roi_f.shape != (60, 40): continue
            score_f = cv2.minMaxLoc(cv2.matchTemplate(roi_f, t['tpl_f'], cv2.TM_CCOEFF_NORMED))[1]
            score_b = cv2.minMaxLoc(cv2.matchTemplate(img[t['y']+30:t['y']+60, t['x']:t['x']+40], t['tpl_b'], cv2.TM_CCOEFF_NORMED))[1]

            best_val = max(score_f, score_b)
            if best_val >= t['thresh']:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': t['id'], 'confidence': round(best_val, 4)})
                break 

        if f_idx % 2500 == 0:
            print(f"  Processed {f_idx}/{len(files)} frames...")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"[DONE] Created: {os.path.basename(OUT_CSV)}")

if __name__ == "__main__":
    run_homing_scan()