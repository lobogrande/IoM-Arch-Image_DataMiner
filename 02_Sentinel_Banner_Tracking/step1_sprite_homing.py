# step1_sprite_homing.py
# Purpose: Step 1 - Spatial detection of mining events using Elastic Homing.
# Version: 3.1 (Universal Elastic Search & Generalized Thresholds)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path() 
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{RUN_ID}.csv")

# --- UNIVERSAL CONSTANTS ---
ORE0_HUD_X, ORE0_HUD_Y = 74, 261
STEP = 59.0
PLAYER_OFFSET = 41.0
TPL_W, TPL_H = 40, 60

# ELASTIC SEARCH: Allows the player to drift +/- 8 pixels without losing the signal
SEARCH_BUFFER = 8 

# GENERALIZED STAIRCASE: Relaxed thresholds to accommodate dataset lighting/compression variance
# Based on Audit averages of ~0.80-0.83
STAIRCASE = {0: 0.80, 1: 0.78, 2: 0.76, 3: 0.74, 4: 0.72, 5: 0.70, 11: 0.78}

def get_slot_geometry(slot_id):
    col = slot_id % 6
    row = slot_id // 6
    hox = int(ORE0_HUD_X + (col * STEP))
    hoy = int(ORE0_HUD_Y + (row * STEP))
    hpx = int(hox - PLAYER_OFFSET) if slot_id < 6 else int(hox + PLAYER_OFFSET)
    apx, apy = int(hpx - (TPL_W // 2)), int(hoy - (TPL_H // 2))
    return (apx, apy)

def run_homing_scan():
    print(f"--- STEP 1: ELASTIC SPRITE HOMING (Target: Run {RUN_ID}) ---")
    full_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    full_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if full_r is None or full_l is None:
        print("Error: Player templates missing.")
        return
    
    bot_r, bot_l = full_r[30:, :], full_l[30:, :]
    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    
    targets =[]
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
        ih, iw = img.shape

        for t in targets:
            # 1. Calculate Elastic Search Window (Base coords + padding)
            x1 = max(0, t['x'] - SEARCH_BUFFER)
            y1 = max(0, t['y'] - SEARCH_BUFFER)
            x2 = min(iw, t['x'] + TPL_W + SEARCH_BUFFER)
            y2 = min(ih, t['y'] + TPL_H + SEARCH_BUFFER)
            
            # Full Body Match
            roi_f = img[y1:y2, x1:x2]
            if roi_f.shape[0] < TPL_H or roi_f.shape[1] < TPL_W: continue
            score_f = cv2.minMaxLoc(cv2.matchTemplate(roi_f, t['tpl_f'], cv2.TM_CCOEFF_NORMED))[1]
            
            # Bottom Half Match
            y1_bot = max(0, t['y'] + 30 - SEARCH_BUFFER)
            roi_b = img[y1_bot:y2, x1:x2]
            if roi_b.shape[0] < 30 or roi_b.shape[1] < TPL_W: continue
            score_b = cv2.minMaxLoc(cv2.matchTemplate(roi_b, t['tpl_b'], cv2.TM_CCOEFF_NORMED))[1]

            best_val = max(score_f, score_b)
            if best_val >= t['thresh']:
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': t['id'], 'confidence': round(best_val, 4)})
                break 

        if f_idx % 2500 == 0:
            print(f"  Processed {f_idx}/{len(files)} frames...")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"[DONE] Total Detections: {len(results)}")
    print(f"Homing map saved to {OUT_CSV}")

if __name__ == "__main__":
    run_homing_scan()