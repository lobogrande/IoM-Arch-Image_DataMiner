# sprite_sequencer.py
# Version: 2.4
# Fix: Robust ROI boundary validation and real-time per-slot hit reporting.

import sys, os, cv2, numpy as np, pandas as pd
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.4"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CALIBRATED GOLDEN CONSTANTS
S0_X, S0_Y = 11, 249
STEP_X, STEP_Y = 117.2, 59.0
WAIT_OFFSET_X = 55.0

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Robust Edge Mode) ---")
    
    # 1. Prepare Edge-Templates
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

    # Define targets
    targets = []
    for s in range(6):
        cx = int(S0_X + (s * STEP_X))
        tx, ty = int(cx - WAIT_OFFSET_X - (tw_r // 2)), int(S0_Y - (th_r // 2))
        targets.append({'slot': s, 'roi': (tx, ty, tw_r + 30, th_r + 20), 'tpl': tpl_r})
    
    cx11 = int(S0_X + (5 * STEP_X))
    tx11, ty11 = int(cx11 + WAIT_OFFSET_X - (tw_l // 2)), int(S0_Y + STEP_Y - (th_l // 2))
    targets.append({'slot': 11, 'roi': (tx11, ty11, tw_l + 30, th_l + 20), 'tpl': tpl_l})

    print(f"Scanning {len(files)} frames...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape
        img_edges = cv2.Canny(img, 100, 200) # Edge detection for UI immunity

        for target in targets:
            tx, ty, tw, th = target['roi']
            t_img = target['tpl']
            
            # Robust Slicing with Boundary Checks
            y1, y2 = max(0, ty), min(ih, ty + th)
            x1, x2 = max(0, tx), min(iw, tx + tw)
            roi = img_edges[y1:y2, x1:x2]

            # CRITICAL FIX: Ensure ROI is strictly larger than template in both dims
            if roi.shape[0] < t_img.shape[0] or roi.shape[1] < t_img.shape[1]:
                continue

            res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)

            if val > 0.28: # Edge matching threshold
                results.append({'frame_idx': f_idx, 'filename': filename, 'slot_id': target['slot'], 'confidence': round(val, 4)})
                counts[target['slot']] += 1
                break

        if f_idx % 1000 == 0:
            # Enhanced Terminal Output: Real-time per-slot hits
            stat_line = " | ".join([f"S{s}:{counts[s]}" for s in sorted(counts.keys())])
            print(f"  [{f_idx:05d}] {stat_line if stat_line else 'Searching...'}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Saved {len(results)} detections. Final Distribution: {dict(counts)}")

if __name__ == "__main__":
    run_sprite_sequencer()