# sprite_y_calibration.py
# Version: 1.3
# Fix: Full-frame search with HUD-safe cropping to prevent size-mismatch crashes.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.3"

def run_y_calibration():
    # 1. Load Left-Facing Template
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_l is None:
        print("Error: player_left.png not found in Reference folder.")
        return
    
    # "Behead" the template (40% crop) to avoid 'Dig Stage' text interference
    th, tw = tpl_l.shape
    tpl_l = tpl_l[int(th*0.4):, :] 
    new_th, new_tw = tpl_l.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith(('.png', '.jpg'))])
    
    # Established Baseline from Slot 0 Audit
    slot0_y_pixel = 249
    
    print(f"--- Y-AXIS RECONCILIATION v{VERSION} (Stabilized) ---")
    print(f"Searching frames 0-1000 for Slot 11 (Start of Floor 13)...")

    best_match = {'conf': 0, 'y': 0, 'frame': '', 'idx': 0}

    # Exhaustive search around index 213 as requested
    for f_idx in range(0, 1000):
        img_path = os.path.join(cfg.get_buffer_path(0), files[f_idx])
        img = cv2.imread(img_path, 0)
        if img is None: continue
        ih, iw = img.shape

        # HUD-SAFE ZONE: Ignore the top 150px (Dig Stage text area)
        # Search the rest of the image
        search_roi = img[150:, :]
        sh, sw = search_roi.shape

        # INTEGRITY CHECK: Ensure search area is larger than template
        if sh < new_th or sw < new_tw:
            continue

        res = cv2.matchTemplate(search_roi, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)

        # We only care about Slot 11, which should be on the right half of the grid
        # and significantly to the right of Slot 0
        if val > best_match['conf'] and loc[0] > (iw // 2):
            best_match = {
                'conf': val,
                'y': loc[1] + 150, # Add back the HUD crop offset
                'frame': files[f_idx],
                'idx': f_idx
            }

        if f_idx % 250 == 0:
            print(f"  Processed {f_idx} frames... (Best Match Conf: {round(best_match['conf'], 3)})")

    if best_match['conf'] > 0.70:
        actual_step_y = best_match['y'] - slot0_y_pixel
        print(f"\n--- Y-CALIBRATION SUCCESS ---")
        print(f"Slot 11 Found at Frame {best_match['idx']} ({best_match['frame']})")
        print(f"Confidence: {round(best_match['conf'], 4)}")
        print(f"Row 2 (Slot 11) Pixel Y: {best_match['y']}")
        print(f"Row 1 (Slot 0)  Pixel Y: {slot0_y_pixel}")
        print(f"CALIBRATED STEP_Y: {actual_step_y}")
        print(f"Old HUD STEP_Y: 59.1")
    else:
        print("\n[ERROR] Still no clear detection at Slot 11.")

if __name__ == "__main__":
    run_y_calibration()