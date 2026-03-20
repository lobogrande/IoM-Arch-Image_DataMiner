# sprite_ultimate_audit.py
# Purpose: Prove 100% grid coverage by fixing the Slot 11 Stand-Offset logic.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- THE FINAL PHYSICS ---
AI_S0_X, AI_S0_Y = 11, 225  # AI Top-Left for Slot 0
STEP = 59.0
STAND_FLIP = 82.0  # Distance from Stand-Left to Stand-Right (41 * 2)

def run_ultimate_audit():
    # Targeted frames including your Frame 213 (F13 Start)
    test_frames = {63: 0, 213: 11, 240: 2, 550: 3}
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    
    print("--- ULTIMATE COORDINATE & SENSITIVITY AUDIT ---")
    
    for f_idx, s_id in test_frames.items():
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]), 0)
        
        # CORRECTED X-LOGIC
        col = s_id % 6
        # AI Top-Left X for Stand-Left
        x_tl = int(AI_S0_X + (col * STEP))
        # If facing left (Slot 11), shift to the Stand-Right position
        if s_id == 11: x_tl += int(STAND_FLIP)
        
        y_tl = int(AI_S0_Y) if s_id < 6 else int(AI_S0_Y + STEP)
        
        # Test Both Methods
        tpl = tpl_r if s_id < 6 else tpl_l
        th, tw = tpl.shape
        
        # 1. Full-Body Score
        roi_f = img[y_tl : y_tl+th, x_tl : x_tl+tw]
        res_f = cv2.matchTemplate(roi_f, tpl, cv2.TM_CCOEFF_NORMED)
        score_f = cv2.minMaxLoc(res_f)[1]
        
        # 2. Bottom-Half Score
        roi_b = img[y_tl+30 : y_tl+th, x_tl : x_tl+tw]
        res_b = cv2.matchTemplate(roi_b, tpl[30:, :], cv2.TM_CCOEFF_NORMED)
        score_b = cv2.minMaxLoc(res_b)[1]

        print(f"Frame {f_idx} (Slot {s_id}):")
        print(f"  Target AI Top-Left: ({x_tl}, {y_tl})")
        print(f"  Full-Body Conf:     {round(score_f, 4)}")
        print(f"  Bottom-Half Conf:   {round(score_b, 4)}")
        print(f"  Winner:             {'Bottom-Half' if score_b > score_f else 'Full-Body'}\n")

if __name__ == "__main__":
    run_ultimate_audit()