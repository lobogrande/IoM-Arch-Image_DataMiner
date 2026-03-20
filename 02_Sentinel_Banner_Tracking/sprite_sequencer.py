# sprite_sequencer.py
# Version: 2.5
# Refactor: 59px Consensus Grid and Per-Slot Terminal Telemetry.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.5"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CONSENSUS COORDINATES
S0_X, S0_Y = 11, 249
STEP_X, STEP_Y = 59.0, 59.0

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Consensus Grid) ---")
    
    # 1. Prepare Edge-Templates for UI Immunity
    def get_edge_tpl(name):
        t = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, name), 0)
        return cv2.Canny(t, 100, 200) if t is not None else None

    tpl_r = get_edge_tpl("player_right.png")
    tpl_l = get_edge_tpl("player_left.png")
    if tpl_r is None or tpl_l is None: return
    th_r, tw_r = tpl_r.shape
    th_l, tw_l = tpl_l.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []
    counts = Counter()

    # 2. Define Targeted Slots (0-5 and 11)
    targets = []
    # Row 1 (Slots 0-5)
    for s in range(6):
        cx = int(S0_X + (s * STEP_X))
        # Box centered on the player's calibrated X for that slot
        tx, ty = cx - (tw_r // 2) - 5, S0_Y - (th_r // 2) - 5
        targets.append({'slot': s, 'roi': (tx, ty, tw_r + 15, th_r + 15), 'tpl': tpl_r})
    
    # Row 2 terminus (Slot 11)
    cx11 = int(S0_X + (5 * STEP_X))
    cy11 = int(S0_Y + STEP_Y)
    tx11, ty11 = cx11 - (tw_l // 2) - 5, cy11 - (th_l // 2) - 5
    targets.append({'slot': 11, 'roi': (tx11, ty11, tw_l + 15, th_l + 15), 'tpl': tpl_l})

    print(f"Scanning {len(files)} frames using 59px grid...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape
        img_edges = cv2.Canny(img, 100, 200)

        for target in targets:
            tx, ty, tw, th = target['roi']
            x1, y1 = max(0, tx), max(0, ty)
            x2, y2 = min(iw, tx + tw), min(ih, ty + th)
            roi = img_edges[y1:y2, x1:x2]

            # Validation: ROI must be large enough for template
            if roi.shape[0] < target['tpl'].shape[0] or roi.shape[1] < target['tpl'].shape[1]:
                continue

            res = cv2.matchTemplate(roi, target['tpl'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)

            if val > 0.28: # Optimized Edge-Threshold
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': target['slot'],
                    'confidence': round(val, 4)
                })
                counts[target['slot']] += 1
                break

        if f_idx % 1000 == 0:
            # Per-Slot terminal reporting
            stat_line = " | ".join([f"S{s}:{counts[s]}" for s in sorted(counts.keys())])
            print(f"  [{f_idx:05d}] {stat_line if stat_line else 'Searching...'}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Final Distribution: {dict(counts)}")

if __name__ == "__main__":
    run_sprite_sequencer()