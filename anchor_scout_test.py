import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/GroundTruth_v72"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]

def run_v72_ground_truth():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- Running v7.2 Ground Truth Scout ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: print(f" [Scanning] Frame {i}...", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. WIDE SEARCH
        search_roi_y1, search_roi_y2 = 200, 350
        search_roi = img_gray[search_roi_y1:search_roi_y2, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > 0.88:
            # Calculate Absolute Coordinates
            abs_x = max_loc[0]
            abs_y = max_loc[1] + search_roi_y1
            
            # DRAW SEARCH BOX (Blue)
            cv2.rectangle(img_bgr, (0, search_roi_y1), (480, search_roi_y2), (255, 0, 0), 1)
            
            # DRAW PLAYER MATCH (Red)
            cv2.rectangle(img_bgr, (abs_x, abs_y), (abs_x + 40, abs_y + 60), (0, 0, 255), 2)
            
            # DRAW VALID ANCHORS (Green Vertical Lines)
            for a in VALID_ANCHORS:
                cv2.line(img_bgr, (a, 200), (a, 350), (0, 255, 0), 1)
            
            # Label with info
            cv2.putText(img_bgr, f"Score: {max_val:.3f} | X: {abs_x} | Y: {abs_y}", (20, 40), 0, 0.6, (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Truth_{i+1:05}.jpg"), img_bgr)
            
            # Stop after 30 images to keep it lean
            if len(os.listdir(OUTPUT_DIR)) > 30:
                print(f"\n[FINISH] Check {OUTPUT_DIR} to see if red box aligns with green lines.")
                return

if __name__ == "__main__":
    run_v72_ground_truth()