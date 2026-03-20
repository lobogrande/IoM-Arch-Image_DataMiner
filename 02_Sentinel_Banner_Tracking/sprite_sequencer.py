# sprite_sequencer.py
# Version: 2.7
# Fix: Implements bottom-weighted matching to bypass UI text in Slots 4 & 5.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.7"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CONSENSUS COORDINATES (AI TOP-LEFT ANCHOR)
S0_X_TL, S0_Y_TL = 11, 225
STEP = 59.0
TPL_W, TPL_H = 40, 60

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Bottom-Half Recovery) ---")
    
    # 1. Prepare Bottom-Half Templates
    # We use the bottom 30px to isolate the feet and shadow from UI text
    full_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    full_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if full_r is None or full_l is None: return
    
    bot_r = full_r[30:, :]
    bot_l = full_l[30:, :]
    bh, bw = bot_r.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    counts = Counter()

    # Define targets for Slots 0-5 and 11
    targets = []
    for s_id in list(range(6)) + [11]:
        # Calculate AI Top-Left for the slot
        col = s_id % 6
        x_tl = int(S0_X_TL + (col * STEP))
        y_tl = int(S0_Y_TL) if s_id < 6 else int(S0_Y_TL + STEP)
        
        tpl = bot_r if s_id < 6 else bot_l
        targets.append({'slot': s_id, 'x': x_tl, 'y': y_tl, 'tpl': tpl})

    print(f"Scanning {len(files)} frames using 0.78 threshold...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        for target in targets:
            # ROI targets the bottom half specifically (y_tl + 30)
            roi_y = target['y'] + 30
            roi_x = target['x']
            
            # Slice the 30x40 area where the feet/shadow should be
            roi = img[roi_y : roi_y + bh, roi_x : roi_x + bw]

            if roi.shape[0] < bh or roi.shape[1] < bw:
                continue

            res = cv2.matchTemplate(roi, target['tpl'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)

            # Applying the 0.78 recovery threshold
            if val >= 0.78:
                # Log HUD center for Step 2 (Match Point + Half-Template Size)
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': target['slot'],
                    'confidence': round(val, 4),
                    'center_x': target['x'] + 20,
                    'center_y': target['y'] + 30
                })
                counts[target['slot']] += 1
                break 

        if f_idx % 2000 == 0:
            stat_line = " | ".join([f"S{s}:{counts[s]}" for s in sorted(counts.keys())])
            print(f"  [{f_idx:05d}] {stat_line if stat_line else 'Searching...'}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Final Distribution: {dict(counts)}")

if __name__ == "__main__":
    run_sprite_sequencer()