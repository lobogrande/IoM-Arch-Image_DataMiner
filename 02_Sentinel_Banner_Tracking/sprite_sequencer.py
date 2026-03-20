# sprite_sequencer.py
# Version: 2.3
# Refactor: Canny Edge Matching to cut through 'Dig Stage' UI and Banners.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.3"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CALIBRATED GOLDEN CONSTANTS
S0_X, S0_Y = 11, 249
STEP_X, STEP_Y = 117.2, 59.0
WAIT_OFFSET_X = 55.0

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Edge-Discovery Mode) ---")
    
    # 1. Prepare Edge-Templates
    def get_edge_tpl(name):
        path = os.path.join(cfg.TEMPLATE_DIR, name)
        t = cv2.imread(path, 0)
        if t is None: return None
        # Apply Canny to find outlines
        edges = cv2.Canny(t, 100, 200)
        return edges

    tpl_r = get_edge_tpl("player_right.png")
    tpl_l = get_edge_tpl("player_left.png")
    if tpl_r is None or tpl_l is None: return

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    counts = Counter()

    # Define targets with the new 117.2 step
    targets = []
    for s in range(6):
        cx = int(S0_X + (s * STEP_X))
        tx, ty = int(cx - WAIT_OFFSET_X - (tpl_r.shape[1]//2)), int(S0_Y - (tpl_r.shape[0]//2))
        targets.append({'slot': s, 'roi': (tx, ty, tpl_r.shape[1]+20, tpl_r.shape[0]+20), 'tpl': tpl_r})
    
    # Slot 11 (Facing Left)
    cx11 = int(S0_X + (5 * STEP_X))
    tx11, ty11 = int(cx11 + WAIT_OFFSET_X - (tpl_l.shape[1]//2)), int(S0_Y + STEP_Y - (tpl_l.shape[0]//2))
    targets.append({'slot': 11, 'roi': (tx11, ty11, tpl_l.shape[1]+20, tpl_l.shape[0]+20), 'tpl': tpl_l})

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        
        # Apply Canny to the frame
        img_edges = cv2.Canny(img, 100, 200)

        for target in targets:
            tx, ty, tw, th = target['roi']
            roi = img_edges[max(0, ty):ty+th, max(0, tx):tx+tw]
            if roi.shape[0] < target['tpl'].shape[0]: continue

            res = cv2.matchTemplate(roi, target['tpl'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)

            # Edge-matching thresholds are typically lower (0.2 - 0.5)
            # but much more distinct from noise.
            if val > 0.25: 
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': target['slot'], 'confidence': round(val, 4)})
                counts[target['slot']] += 1
                break

        if f_idx % 2000 == 0:
            stat_line = " ".join([f"S{s}:{c}" for s, c in sorted(counts.items())])
            print(f"  [Progress] {f_idx}/{len(files)} | {stat_line}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(results)} detections. Final: {dict(counts)}")

if __name__ == "__main__":
    run_sprite_sequencer()