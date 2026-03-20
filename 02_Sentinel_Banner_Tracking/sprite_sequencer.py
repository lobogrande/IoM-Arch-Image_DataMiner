# sprite_sequencer.py
# Version: 2.6
# Fix: Implements the 59px/41px Consensus Grid with Decoupled AI/HUD coordinates.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.6"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CONSENSUS COORDINATES (From v1.1 Verification)
ORE0_CENTER_X, ORE0_CENTER_Y = 72, 255
STEP = 59.0
OFFSET = 41.0
TPL_W, TPL_H = 40, 60

def get_player_data(slot_id):
    """Returns (visual_center_x, visual_center_y, ai_tl_x, ai_tl_y)"""
    col = slot_id % 6
    row = 0 if slot_id < 6 else 1
    
    # Calculate HUD Center
    ore_x = ORE0_CENTER_X + (col * STEP)
    ore_y = ORE0_CENTER_Y + (row * STEP)
    # Stand position (Right of ore for S0-5, Left for S11)
    cx = (ore_x - OFFSET) if slot_id < 6 else (ore_x + OFFSET)
    cy = ore_y
    
    # Calculate AI Top-Left (20px left, 30px up from center)
    tl_x, tl_y = cx - (TPL_W // 2), cy - (TPL_H // 2)
    
    return (int(cx), int(cy)), (int(tl_x), int(tl_y))

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Consensus Integration) ---")
    
    # 1. Load Templates
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_r is None or tpl_l is None: return

    # 2. Setup Slot Thresholds (Lower for noisy center slots)
    # Slots 2 and 3 are historically noisy due to UI text.
    thresholds = {s: 0.84 for s in range(6)}
    thresholds[2] = 0.76
    thresholds[3] = 0.76
    thresholds[11] = 0.80

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    counts = Counter()

    # Pre-calculate search boxes for all 7 slots
    targets = []
    for s_id in list(range(6)) + [11]:
        center, tl = get_player_data(s_id)
        # Search ROI: 15px padding around the 40x60 sprite
        roi = (tl[0]-5, tl[1]-5, TPL_W+10, TPL_H+10)
        tpl = tpl_r if s_id < 6 else tpl_l
        targets.append({'slot': s_id, 'roi': roi, 'tpl': tpl, 'thresh': thresholds.get(s_id, 0.82)})

    print(f"Scanning {len(files)} frames with surgical ROIs...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        for target in targets:
            tx, ty, tw, th = target['roi']
            x1, y1, x2, y2 = max(0, tx), max(0, ty), min(iw, tx+tw), min(ih, ty+th)
            roi = img[y1:y2, x1:x2]

            if roi.shape[0] < TPL_H or roi.shape[1] < TPL_W:
                continue

            res = cv2.matchTemplate(roi, target['tpl'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)

            if val >= target['thresh']:
                # Log the visual center for Step 2 downstream
                vis_center, _ = get_player_data(target['slot'])
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': target['slot'],
                    'confidence': round(val, 4),
                    'center_x': vis_center[0],
                    'center_y': vis_center[1]
                })
                counts[target['slot']] += 1
                break # Only one detection per frame

        if f_idx % 2000 == 0:
            stat_line = " | ".join([f"S{s}:{counts[s]}" for s in sorted(counts.keys())])
            print(f"  [{f_idx:05d}] {stat_line if stat_line else 'Searching...'}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(results)} detections. Final Map: {dict(counts)}")

if __name__ == "__main__":
    run_sprite_sequencer()