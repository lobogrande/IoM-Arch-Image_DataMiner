# sprite_dual_calibration.py
# Purpose: Finalize the disparity between AI Top-Left and HUD Center coordinates.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_dual_calibration():
    # We use the two anchors: Slot 0 (Right) and Slot 11 (Left)
    test_files = [
        {"idx": 63, "label": "Slot 0 (Facing Right)", "tpl": "player_right.png"},
        {"idx": 213, "label": "Slot 11 (Facing Left)", "tpl": "player_left.png"}
    ]
    
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    print("--- DUAL COORDINATE CALIBRATION ---")
    
    for item in test_files:
        img_gray = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[item['idx']]), 0)
        tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, item['tpl']), 0)
        th, tw = tpl.shape

        # 1. FIND AI TOP-LEFT (The "Anchor")
        # Scan a generous area around the player
        res = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        
        # 2. CALCULATE HUD CENTER
        # If the template matches perfectly at 'loc', the center is:
        center_x = loc[0] + (tw // 2)
        center_y = loc[1] + (th // 2)

        print(f"\nRESULTS FOR {item['label']}:")
        print(f"  AI Top-Left (Match Point): X={loc[0]}, Y={loc[1]} (Conf: {round(val,4)})")
        print(f"  HUD Visual Center:        X={center_x}, Y={center_y}")
        print(f"  Template Dimensions:      {tw}w x {th}h")

if __name__ == "__main__":
    run_dual_calibration()