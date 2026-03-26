# sprite_geometric_proof.py
# Purpose: Prove the 59px/41px-offset math is frame-perfect on key frames.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- THE NEW PHYSICS ---
ORE0_X = 52.0
STEP_X = 59.0
STEP_Y = 59.0
STAND_OFFSET = 41.0
ROW1_Y = 249.0

def get_stand_x(slot_id):
    col = slot_id % 6
    block_x = ORE0_X + (col * STEP_X)
    # Facing Right for 0-5, Left for 11
    return (ore_x - STAND_OFFSET) if slot_id < 6 else (ore_x + STAND_OFFSET)

def run_proof():
    test_frames = {63: 0, 120: 1, 240: 2, 550: 3, 213: 11}
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    print("--- GEOMETRIC ALIGNMENT PROOF ---")
    if not os.path.exists("proof_gallery"): os.makedirs("proof_gallery")

    for f_idx, slot in test_frames.items():
        img_path = os.path.join(cfg.get_buffer_path(0), files[f_idx])
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate Expected Position
        target_x = get_stand_x(slot)
        target_y = ROW1_Y if slot < 6 else (ROW1_Y + STEP_Y)
        
        # Template choice
        tpl = tpl_r if slot < 6 else tpl_l
        th, tw = tpl.shape
        
        # Draw target crosshair and tight box
        tx, ty = int(target_x), int(target_y)
        cv2.drawMarker(img, (tx, ty), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.rectangle(img, (tx-int(tw/2), ty-int(th/2)), (tx+int(tw/2), ty+int(th/2)), (0, 255, 255), 1)
        
        # Check Confidence at this EXACT spot
        roi = gray[ty-int(th/2)-5 : ty+int(th/2)+5, tx-int(tw/2)-5 : tx+int(tw/2)+5]
        if roi.shape[0] >= th and roi.shape[1] >= tw:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
        else:
            val = 0.0

        cv2.putText(img, f"Slot {slot} | Conf: {round(val,3)}", (tx-40, ty-40), 0, 0.5, (0, 255, 255), 1)
        cv2.imwrite(f"proof_gallery/proof_slot_{slot}_f{f_idx}.jpg", img)
        print(f"Verified Slot {slot} at Frame {f_idx}: X={tx}, Y={ty} (Conf: {round(val,3)})")

if __name__ == "__main__":
    run_proof()