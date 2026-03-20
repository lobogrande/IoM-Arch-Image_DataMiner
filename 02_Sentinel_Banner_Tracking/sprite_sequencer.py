# sprite_sequencer.py
# Version: 1.9
# Refactor: Calibrated Golden Steps (118x, 59y) with Headless ROI scanning.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.9"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CALIBRATED GOLDEN CONSTANTS
PIXEL_GRID_X_START = 66.0  # AI Anchor for Slot 0 center
PIXEL_GRID_Y_START = 249.0 # AI Anchor for Row 1
STEP_X = 118.0
STEP_Y = 59.0
WAIT_OFFSET_X = 55.0 

def get_pixel_coords(slot_id):
    row = slot_id // 6
    col = slot_id % 6
    x = int(PIXEL_GRID_X_START + (col * STEP_X))
    y = int(PIXEL_GRID_Y_START + (row * STEP_Y))
    return x, y

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Golden Master) ---")
    
    # 1. Load and Behead Templates (Crop top 40% for UI immunity)
    raw_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    raw_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if raw_r is None or raw_l is None: return

    tpl_r = raw_r[int(raw_r.shape[0]*0.4):, :]
    tpl_l = raw_l[int(raw_l.shape[0]*0.4):, :]
    h_r, w_r = tpl_r.shape
    h_l, w_l = tpl_l.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    # 2. Setup Targeted ROIs
    targets = []
    # Slots 0-5 (Facing Right, Sits Left of Slot)
    for s_id in range(6):
        cx, cy = get_pixel_coords(s_id)
        tx = int(cx - WAIT_OFFSET_X - (w_r // 2))
        ty = int(cy - (h_r // 2))
        targets.append({'slot': s_id, 'roi': (tx, ty, w_r+10, h_r+10), 'tpl': tpl_r})

    # Slot 11 (Facing Left, Sits Right of Slot)
    cx11, cy11 = get_pixel_coords(11)
    tx11 = int(cx11 + WAIT_OFFSET_X - (w_l // 2))
    ty11 = int(cy11 - (h_l // 2))
    targets.append({'slot': 11, 'roi': (tx11, ty11, w_l+10, h_l+10), 'tpl': tpl_l})

    print(f"Scanning {len(files)} frames using calibrated grid...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        for target in targets:
            tx, ty, tw, th = target['roi']
            # Safety Clamp
            x1, y1 = max(0, tx), max(0, ty)
            x2, y2 = min(iw, tx + tw), min(ih, ty + th)
            
            roi = img[y1:y2, x1:x2]
            if roi.shape[0] < target['tpl'].shape[0] or roi.shape[1] < target['tpl'].shape[1]:
                continue

            res = cv2.matchTemplate(roi, target['tpl'], cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > 0.85:
                results.append({
                    'frame_idx': f_idx,
                    'filename': filename,
                    'slot_id': target['slot'],
                    'confidence': round(max_val, 4)
                })
                break 

        if f_idx % 2000 == 0:
            print(f"  [Progress] {f_idx} frames... (Hits: {len(results)})")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] v{VERSION} saved {len(results)} detections to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()