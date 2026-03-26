# sprite_visual_overlay.py
# Purpose: Draw the 7-slot grid search boxes onto raw frames for visual verification.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CALIBRATED CONSTANTS
S0_X, S0_Y = 11, 249
STEP_X, STEP_Y = 117.2, 59.0

def run_visual_overlay():
    # Pick a few key frames where we expect movement (e.g. start of Floor 13)
    test_frames = [100, 200, 300, 400, 500, 550] 
    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    
    if not os.path.exists("debug_grid"): os.makedirs("debug_grid")

    print("--- GENERATING VISUAL GRID OVERLAY ---")

    for f_idx in test_frames:
        img_path = os.path.join(cfg.get_buffer_path(0), files[f_idx])
        img = cv2.imread(img_path)
        if img is None: continue

        # Draw the "Waiting Boxes" we are using in the sequencer
        for s in range(6):
            cx = int(S0_X + (s * STEP_X))
            # The box we used in v2.4 (centered on player)
            x1, y1 = cx - 10, S0_Y - 20
            cv2.rectangle(img, (x1, y1), (x1+60, y1+80), (0, 255, 0), 2)
            cv2.putText(img, f"Slot {s}", (x1, y1-5), 0, 0.5, (0, 255, 0), 1)

        # Slot 11
        cx11 = int(S0_X + (5 * STEP_X))
        x11, y11 = cx11 - 10, int(S0_Y + STEP_Y) - 20
        cv2.rectangle(img, (x11, y11), (x11+60, y11+80), (255, 0, 0), 2)
        cv2.putText(img, "Slot 11", (x11, y11-5), 0, 0.5, (255, 0, 0), 1)

        out_name = f"debug_grid/grid_verify_frame_{f_idx}.jpg"
        cv2.imwrite(out_name, img)
        print(f"Saved: {out_name}")

if __name__ == "__main__":
    run_visual_overlay()