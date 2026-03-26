# sprite_consensus_verification.py
# Version: 1.1
# Fix: Corrected 20px/30px offset and resized HUD boxes to wrap player dimensions.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- FINAL CALIBRATED CONSTANTS ---
ORE0_CENTER_HUD = (72, 255) # The actual pixel center of the first block
STEP = 59.0
OFFSET = 41.0
TPL_W, TPL_H = 40, 60

def get_player_center(slot_id):
    col = slot_id % 6
    row = 0 if slot_id < 6 else 1
    # Center of the Block
    ocx = ORE0_CENTER_HUD[0] + (col * STEP)
    ocy = ORE0_CENTER_HUD[1] + (row * STEP)
    # Stand position (Left of block for S0-5, Right for S11)
    scx = (ocx - OFFSET) if slot_id < 6 else (ocx + OFFSET)
    return int(scx), int(ocy)

def run_verification():
    # Focused audit on the 4 confirmed valid frames
    test_frames = {63: 0, 240: 2, 550: 3, 213: 11}
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    print("--- CONSENSUS ALIGNMENT VERIFICATION v1.1 ---")
    if not os.path.exists("consensus_proof"): os.makedirs("consensus_proof")

    for f_idx, slot in test_frames.items():
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Calculate HUD Center (Torso)
        cx, cy = get_player_center(slot)
        
        # 2. Derive AI Top-Left Match Point
        tl_x, tl_y = cx - (TPL_W // 2), cy - (TPL_H // 2)
        
        # Draw HUD Box (Yellow) - Resized to match player 40x60 dimensions
        cv2.rectangle(img, (tl_x, tl_y), (tl_x + TPL_W, tl_y + TPL_H), (0, 255, 255), 1)
        # Draw HUD Center (Purple Dot)
        cv2.circle(img, (cx, cy), 2, (255, 0, 255), -1)
        # Draw AI Match Point Crosshair (Green)
        cv2.drawMarker(img, (tl_x, tl_y), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # Check Confidence at THIS EXACT AI Match Point
        roi = gray[max(0, tl_y-2):tl_y+TPL_H+2, max(0, tl_x-2):tl_x+TPL_W+2]
        tpl = tpl_r if slot < 6 else tpl_l
        if roi.shape[0] >= TPL_H and roi.shape[1] >= TPL_W:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            conf = round(val, 4)
        else: conf = 0.0

        cv2.putText(img, f"Slot {slot} Conf: {conf}", (cx-40, cy-40), 0, 0.4, (0, 255, 255), 1)
        cv2.imwrite(f"consensus_proof/verify_slot_{slot}.jpg", img)
        print(f"Verified Slot {slot} (Frame {f_idx}): Conf {conf} | Center:({cx},{cy})")

if __name__ == "__main__":
    run_verification()