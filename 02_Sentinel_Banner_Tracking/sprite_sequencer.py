# sprite_sequencer.py
# Version: 2.1
# Fix: Reverted to Full Template with Masking to bypass UI text while keeping high confidence.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "2.1"
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"sprite_homing_run_{BUFFER_ID}.csv")

# CALIBRATED PIXEL CONSTANTS (From Audits)
SPRITE_0_CENTER = (11, 249) # Absolute AI Pixel location of sprite at Slot 0
STEP_X = 118.0
STEP_Y = 59.0

# Detection Settings
MATCH_THRESHOLD = 0.82 # Higher threshold now possible with full-template masking

def get_sprite_center(slot_id):
    row = slot_id // 6
    col = slot_id % 6
    x = int(SPRITE_0_CENTER[0] + (col * STEP_X))
    y = int(SPRITE_0_CENTER[1] + (row * STEP_Y))
    return x, y

def create_ui_mask(template):
    """Creates a mask that ignores the top 40% (UI text area) while keeping the body."""
    mask = np.ones(template.shape, dtype=np.float32)
    # Set the top 40% to 0 (ignored in the match calculation)
    mask[0:int(template.shape[0]*0.4), :] = 0.0
    return mask

def run_sprite_sequencer():
    print(f"--- SPRITE SEQUENCER v{VERSION} (Masked Production) ---")
    
    # 1. Load Full Templates
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_r is None or tpl_l is None: return

    # 2. Create UI-Immunity Masks
    mask_r = create_ui_mask(tpl_r)
    mask_l = create_ui_mask(tpl_l)
    
    h_r, w_r = tpl_r.shape
    h_l, w_l = tpl_l.shape

    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])
    results = []

    # 3. Setup Calibrated Targets
    targets = []
    # Slots 0-5 (Facing Right)
    for s_id in range(6):
        cx, cy = get_sprite_center(s_id)
        # Search window is 100px wide to account for idle movement
        targets.append({'slot': s_id, 'roi_box': (cx-50, cy-40, 100, 80), 'tpl': tpl_r, 'mask': mask_r})

    # Slot 11 (Facing Left)
    cx11, cy11 = get_sprite_center(11)
    targets.append({'slot': 11, 'roi_box': (cx11-50, cy11-40, 100, 80), 'tpl': tpl_l, 'mask': mask_l})

    print(f"Scanning {len(files)} frames...")

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue
        ih, iw = img.shape

        for target in targets:
            tx, ty, tw, th = target['roi_box']
            x1, y1 = max(0, tx), max(0, ty)
            x2, y2 = min(iw, tx + tw), min(ih, ty + th)
            
            roi = img[y1:y2, x1:x2]
            # Ensure ROI is at least as big as template
            if roi.shape[0] < target['tpl'].shape[0] or roi.shape[1] < target['tpl'].shape[1]:
                continue

            # Masked Template Matching (Requires TM_SQDIFF or TM_CCORR in older CV, 
            # but TM_CCOEFF_NORMED is supported in newer versions)
            res = cv2.matchTemplate(roi, target['tpl'], cv2.TM_CCOEFF_NORMED, mask=target['mask'])
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val >= MATCH_THRESHOLD:
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
    print(f"\n[DONE] v{VERSION} saved {len(results)} hits to {OUT_CSV}")

if __name__ == "__main__":
    run_sprite_sequencer()