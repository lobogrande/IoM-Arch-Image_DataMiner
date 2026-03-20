# sprite_y_calibration.py
# Version: 1.0
# Purpose: Calibrate STEP_Y by finding the player at Slot 11.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_y_calibration():
    # 1. Load Left-Facing Template (Behead for stability)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_l is None: return
    tpl_l = tpl_l[int(tpl_l.shape[0]*0.4):, :] 

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    # Slot 0 Y was confirmed as 249 in our previous audit
    slot0_y_pixel = 249
    
    print("--- STARTING Y-AXIS RECONCILIATION ---")
    print("Searching for player at Slot 11 (Facing Left)...")

    found_y = None
    # We scan a broader range of frames to find the first Floor 13 transition
    for f_idx in range(1000, 8000):
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]), 0)
        if img is None: continue

        # Global search for the left-facing sprite in the second row area
        # We look in the right half of the screen (x > 800)
        res = cv2.matchTemplate(img[:, 800:], tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)

        if val > 0.88:
            found_y = loc[1]
            print(f"Found Slot 11 at Frame {f_idx}: Pixel Y = {found_y} (Conf: {round(val,4)})")
            break

    if found_y:
        # Step_Y is the distance between Row 1 (Slot 0) and Row 2 (Slot 11)
        actual_step_y = found_y - slot0_y_pixel
        print(f"\n--- Y-CALIBRATION RESULT ---")
        print(f"Row 1 (Slot 0) AI Pixel Y: {slot0_y_pixel}")
        print(f"Row 2 (Slot 11) AI Pixel Y: {found_y}")
        print(f"NEW CALIBRATED STEP_Y: {actual_step_y}")
        print(f"Old HUD STEP_Y: 59.1")
    else:
        print("\n[ERROR] Could not find player at Slot 11. Try increasing the search range.")

if __name__ == "__main__":
    run_y_calibration()