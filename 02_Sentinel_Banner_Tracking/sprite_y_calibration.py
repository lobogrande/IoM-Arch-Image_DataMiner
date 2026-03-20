# sprite_y_calibration.py
# Version: 1.1
# Fix: Full-width search and dimension validation to prevent OpenCV crashes.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.1"

def run_y_calibration():
    # 1. Load Left-Facing Template (Behead for stability)
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_l is None:
        print("Error: player_left.png not found.")
        return
    
    # Crop top 40% to avoid 'Dig Stage' text interference
    tpl_l = tpl_l[int(tpl_l.shape[0]*0.4):, :] 
    th, tw = tpl_l.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    # Slot 0 Y was confirmed as 249 in our previous audit
    slot0_y_pixel = 249
    
    print(f"--- Y-AXIS RECONCILIATION v{VERSION} ---")
    print("Searching for player at Slot 11 (Facing Left)...")

    found_y = None
    # We scan a broad range of frames to find where the player transitions to Row 2
    # Typically this happens toward the end of a floor cycle
    for f_idx in range(100, len(files), 10): # Sample every 10 frames to move fast
        img = cv2.imread(os.path.join(cfg.get_buffer_path(0), files[f_idx]), 0)
        if img is None: continue
        ih, iw = img.shape

        # Safety Check: Ensure image is large enough for the template
        if ih < th or iw < tw: continue

        # Global search for the left-facing sprite
        # We look across the entire width now to avoid slicing errors
        res = cv2.matchTemplate(img, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)

        # We only care about hits on the right half of the grid (X > 400)
        # to ensure we are actually looking at Slot 11 and not a stray fairy
        if val > 0.90 and loc[0] > 400:
            found_y = loc[1]
            print(f"Found Slot 11 candidate at Frame {f_idx}:")
            print(f"  Pixel X: {loc[0]}, Pixel Y: {found_y} (Conf: {round(val,4)})")
            break

    if found_y:
        actual_step_y = found_y - slot0_y_pixel
        print(f"\n--- Y-CALIBRATION RESULT ---")
        print(f"Row 1 (Slot 0)  AI Pixel Y: {slot0_y_pixel}")
        print(f"Row 2 (Slot 11) AI Pixel Y: {found_y}")
        print(f"NEW CALIBRATED STEP_Y: {actual_step_y}")
        print(f"Old HUD STEP_Y: 59.1")
    else:
        print("\n[ERROR] No Slot 11 detections. Check if the player actually reaches Slot 11 in this buffer.")

if __name__ == "__main__":
    run_y_calibration()