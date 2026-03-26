import sys, os
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- DIAGNOSTIC CONFIG ---
BUFFER_ID = 0  
SOURCE_DIR = cfg.get_buffer_path(BUFFER_ID)
DEBUG_DIR = "debug_sprite_boxes"
if not os.path.exists(DEBUG_DIR): os.makedirs(DEBUG_DIR)

# Updated to use the 74 start found in your other scripts
GRID_X_START = 74 
GRID_Y_START = 261
STEP_X = 107.5
STEP_Y = 59.1
WAIT_OFFSET_X = 55 

def run_diagnostic():
    print("--- STARTING SPRITE SEARCH DIAGNOSTIC ---")
    
    tpl_r = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "player_right.png"), 0)
    if tpl_r is None:
        print("Error: player_right.png not found!")
        return

    h_r, w_r = tpl_r.shape
    files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith('.png')])[:500]
    
    # We'll just test Slot 0 (The most likely to fail due to edge clamping)
    cx, cy = GRID_X_START, GRID_Y_START
    tw, th = w_r + 20, h_r + 20 # Give it a larger 20px padding for diagnostics
    tx = int(np.clip(cx - WAIT_OFFSET_X - (tw // 2), 0, 1920 - tw))
    ty = int(np.clip(cy - (th // 2), 0, 1080 - th))

    print(f"Testing Slot 0 ROI at: x={tx}, y={ty}, w={tw}, h={th}")
    
    best_overall_conf = 0

    for f_idx, filename in enumerate(files):
        img = cv2.imread(os.path.join(SOURCE_DIR, filename), 0)
        if img is None: continue

        roi = img[ty:ty+th, tx:tx+tw]
        res = cv2.matchTemplate(roi, tpl_r, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_overall_conf:
            best_overall_conf = max_val
            # Save the "best" match image so we can see it
            cv2.imwrite(os.path.join(DEBUG_DIR, f"best_match_frame_{f_idx}.png"), roi)

        if f_idx % 100 == 0:
            print(f"  Frame {f_idx}: Best Match so far: {round(best_overall_conf, 4)}")

    print(f"\n--- DIAGNOSTIC COMPLETE ---")
    print(f"Highest confidence found in 500 frames: {round(best_overall_conf, 4)}")
    print(f"Check the '{DEBUG_DIR}' folder to see if the sprite is actually in the box.")

if __name__ == "__main__":
    run_diagnostic()