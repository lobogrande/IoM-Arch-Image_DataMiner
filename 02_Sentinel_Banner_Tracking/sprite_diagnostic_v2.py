# sprite_diagnostic_v2.py
# Purpose: Identify why Slots 2-5 and 11 are failing by saving 'Best Match' ROIs.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DIAG_DIR = "forensic_sprite_audit"
if not os.path.exists(DIAG_DIR): os.makedirs(DIAG_DIR)

# Constants from your calibrated scripts
GRID_X_START, GRID_Y_START = 74, 261
STEP_X, STEP_Y = 107.5, 59.1
OFFSETS_TO_TEST = [50, 55, 60]

def run_forensic_audit():
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    # MASK: Create a mask that ignores the top 35% of the template 
    # (where 'Dig Stage' text usually interferes)
    mask_r = np.ones(tpl_r.shape, dtype=np.uint8) * 255
    mask_r[0:int(tpl_r.shape[0]*0.35), :] = 0 

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])[:3000]
    
    # We will track the BEST image seen for EVERY slot
    best_results = {s: {'conf': 0, 'img': None, 'frame': ''} for s in [0,1,2,3,4,5,11]}

    print("--- STARTING FORENSIC SPRITE AUDIT ---")
    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), filename), 0)
        if img is None: continue

        for slot_id in best_results.keys():
            # Calculate coordinates
            row, col = slot_id // 6, slot_id % 6
            cx, cy = int(GRID_X_START + (col * STEP_X)), int(GRID_Y_START + (row * STEP_Y))
            
            # Determine template and side
            tpl = tpl_r if slot_id <= 5 else tpl_l
            side_mult = -1 if slot_id <= 5 else 1
            
            for off in OFFSETS_TO_TEST:
                tx = int(cx + (side_mult * off) - (tpl.shape[1]//2))
                ty = int(cy - (tpl.shape[0]//2))
                
                # Extract ROI (with safety clamping)
                roi = img[max(0, ty):ty+tpl.shape[0], max(0, tx):tx+tpl.shape[1]]
                if roi.shape != tpl.shape: continue

                # Match with Masking
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED, mask=mask_r if slot_id <= 5 else None)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > best_results[slot_id]['conf']:
                    best_results[slot_id]['conf'] = max_val
                    best_results[slot_id]['img'] = roi
                    best_results[slot_id]['frame'] = filename

    # Save the 'Hall of Fame' images for each slot
    for s_id, data in best_results.items():
        if data['img'] is not None:
            out_name = f"slot_{s_id}_conf_{round(data['conf'], 3)}_{data['frame']}"
            cv2.imwrite(os.path.join(DIAG_DIR, out_name), data['img'])
            print(f"Slot {s_id}: Best match {data['conf']} found in {data['frame']}")

if __name__ == "__main__":
    run_forensic_audit()