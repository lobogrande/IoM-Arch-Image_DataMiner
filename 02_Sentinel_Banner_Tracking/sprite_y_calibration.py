# sprite_y_calibration.py
# Version: 1.2
# Fix: Exhaustive search (every frame) and lower threshold to capture transition frames.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

VERSION = "1.2"

def run_y_calibration():
    # 1. Load Left-Facing Template
    tpl_l = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_left.png"), 0)
    if tpl_l is None: return
    
    # Behead for stability (40% crop)
    th, tw = tpl_l.shape
    tpl_l = tpl_l[int(th*0.4):, :] 
    new_th, _ = tpl_l.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    # Established Baseline from Slot 0 Audit
    slot0_y_pixel = 249
    
    print(f"--- Y-AXIS RECONCILIATION v{VERSION} (Exhaustive Mode) ---")
    print(f"Searching every frame for Slot 11 (Start of Floor 13)...")

    best_match = {'conf': 0, 'y': 0, 'frame': '', 'idx': 0}

    # Focus specifically on the early transition window the user identified
    for f_idx in range(0, 1000):
        img_path = os.path.join(cfg.get_buffer_path(0), files[f_idx])
        img = cv2.imread(img_path, 0)
        if img is None: continue
        ih, iw = img.shape

        # Search the right half where Slot 11 lives (X > 500)
        # Full height scan to find the vertical shift
        strip = img[:, 500:]
        res = cv2.matchTemplate(strip, tpl_l, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)

        if val > best_match['conf']:
            best_match = {
                'conf': val,
                'y': loc[1],
                'frame': files[f_idx],
                'idx': f_idx
            }

        # Progress heartbeat
        if f_idx % 200 == 0:
            print(f"  Processed {f_idx} frames... (Best so far: {round(best_match['conf'], 3)})")

    if best_match['conf'] > 0.75:
        actual_step_y = best_match['y'] - slot0_y_pixel
        print(f"\n--- Y-CALIBRATION SUCCESS ---")
        print(f"Best Slot 11 Candidate: Frame {best_match['idx']} ({best_match['frame']})")
        print(f"Confidence: {round(best_match['conf'], 4)}")
        print(f"Row 2 (Slot 11) Pixel Y: {best_match['y']}")
        print(f"Row 1 (Slot 0)  Pixel Y: {slot0_y_pixel}")
        print(f"CALIBRATED STEP_Y: {actual_step_y}")
        print(f"Old HUD STEP_Y: 59.1")
    else:
        print("\n[ERROR] Still no clear detection. Template/Search mismatch.")

if __name__ == "__main__":
    run_y_calibration()