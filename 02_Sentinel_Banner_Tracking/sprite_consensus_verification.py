# sprite_consensus_verification.py
# Purpose: Final proof of the 59px consensus grid with AI/HUD decoupled coordinates.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONSENSUS CONSTANTS
ORE0_CENTER = (52, 255)
STEP = 59.0
OFFSET = 41.0

def get_coords(slot_id):
    col = slot_id % 6
    row = 0 if slot_id < 6 else 1
    # Center of the Ore
    ocx = ORE0_CENTER[0] + (col * STEP)
    ocy = ORE0_CENTER[1] + (row * STEP)
    # Stand position (Right of ore for S0-5, Left for S11)
    scx = (ocx - OFFSET) if slot_id < 6 else (ocx + OFFSET)
    return int(scx), int(ocy)

def run_verification():
    test_frames = {63: 0, 213: 11, 240: 2, 550: 3}
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    print("--- CONSENSUS ALIGNMENT VERIFICATION ---")
    if not os.path.exists("consensus_proof"): os.makedirs("consensus_proof")

    for f_idx, slot in test_frames.items():
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. HUD Center
        cx, cy = get_coords(slot)
        # 2. AI Top-Left (Match Point)
        th, tw = (60, 40) # From calibration output
        tl_x, tl_y = cx - 20, cy - 30
        
        # Draw HUD Box (Yellow)
        cv2.rectangle(img, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 255), 1)
        # Draw AI Match Point (Green)
        cv2.drawMarker(img, (tl_x, tl_y), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # Check Confidence at AI TL
        roi = gray[tl_y-2:tl_y+th+2, tl_x-2:tl_x+tw+2]
        tpl = tpl_r if slot < 6 else tpl_l
        if roi.shape[0] >= th and roi.shape[1] >= tw:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            conf = round(val, 4)
        else: conf = 0.0

        cv2.putText(img, f"Slot {slot} Conf: {conf}", (cx-40, cy-35), 0, 0.4, (0, 255, 255), 1)
        cv2.imwrite(f"consensus_proof/verify_slot_{slot}.jpg", img)
        print(f"Frame {f_idx} (Slot {slot}): Confidence {conf} at AI:({tl_x},{tl_y}) HUD:({cx},{cy})")

if __name__ == "__main__":
    run_verification()