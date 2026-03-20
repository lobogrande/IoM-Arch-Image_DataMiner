# sprite_step_calibration.py
# Purpose: Verify the '118 Pixel Step' theory to fix Slot 5 blindness.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

def run_step_calibration():
    tpl = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    tpl = tpl[int(tpl.shape[0]*0.4):, :] # Behead
    
    # We'll use two frames where we know the player is at Slot 0 and Slot 1/2
    # Based on your CSV: Frame 64 (Slot 0) and Frame 24398 (Slot 2)
    test_frames = [64, 24398]
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    coords = []
    for f_idx in test_frames:
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]), 0)
        res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        coords.append(loc[0])
        print(f"Frame {f_idx}: Detected Sprite at Pixel X = {loc[0]} (Conf: {round(val,3)})")

    if len(coords) == 2:
        # If Frame 24398 is Slot 2, the distance should be 2 * Step
        total_dist = coords[1] - coords[0]
        # We need to know if Frame 24398 is Slot 1 or Slot 2 to finalize the math.
        # Looking at your CSV, 24398 was labeled 'Slot 2'
        inferred_step = total_dist / 2 
        print(f"\n--- CALIBRATION RESULT ---")
        print(f"Measured Distance: {total_dist} pixels across 2 slots.")
        print(f"NEW CALIBRATED STEP_X: {inferred_step}")
        print(f"Old HUD STEP_X: 107.5")

if __name__ == "__main__":
    run_step_calibration()