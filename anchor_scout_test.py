import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/TargetScout_v67"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.85  
OFFSET_X = 24           

def run_v67_visual_scout():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # Floor 1
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Target_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v6.7 Visual Target Scout ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: print(f" [Scanning] Frame {i}...", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. FIND PLAYER
        search_roi = img_gray[230:310, 0:450]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. CALCULATE SAMPLE POINT
            p_center_x = max_loc[0] + 20
            # Target Ore Center
            target_x = int(p_center_x + OFFSET_X)
            target_y = 261 
            
            # 3. DRAW VISUAL MARKERS
            # Red Circle: Where we think the player center is
            cv2.circle(img_bgr, (p_center_x, max_loc[1] + 230 + 30), 3, (0, 0, 255), -1)
            # Green Crosshair: Where we are sampling the ORE DNA
            cv2.drawMarker(img_bgr, (target_x, target_y), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
            
            # Label with score
            cv2.putText(img_bgr, f"Score: {max_val:.2f} | X:{target_x}", (30, 50), 0, 0.7, (0, 255, 0), 2)
            
            # Save every time we find a player match to see why it fails
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Target_{i+1:05}.jpg"), img_bgr)
            
            # Stop after finding a few examples to keep it fast
            if len(os.listdir(OUTPUT_DIR)) > 20:
                print(f"\n[FINISH] 20 debug frames generated in {OUTPUT_DIR}")
                return

if __name__ == "__main__":
    run_v67_visual_scout()