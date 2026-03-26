import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
# Surgical Constants - Loosened for extreme debugging
MATCH_THRESHOLD = 0.80 
OFFSET_X = 24           

def run_v63_forensic_heartbeat():
    player_t = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- Running v6.3 Forensic Heartbeat ---")

    for i in range(len(buffer_files) - 1):
        # We only care about the first 500 frames for this test
        if i > 500: break

        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_gray is None: continue

        # 1. TEMPLATE MATCH
        # Widening the net: Y from 200 to 400
        search_roi = img_gray[200:400, 0:450]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. CALCULATE ABSOLUTE COORDINATES
            # max_loc is relative to the ROI (which starts at Y=200)
            abs_player_x = max_loc[0]
            abs_player_y = max_loc[1] + 200
            
            player_center_x = abs_player_x + 20 # Half of 40px template
            expected_block_x = player_center_x + OFFSET_X
            
            # 3. TEST DNA AT EXPECTED OFFSET
            # We are going to sample the exact point the script "thinks" the block is
            sample_roi = img_gray[261-5:261+5, int(expected_block_x)-5:int(expected_block_x)+5]
            diff = np.sum(cv2.absdiff(sample_roi, bg_t[0][19:29, 19:29])) / 100
            
            # LOG EVERYTHING TO CONSOLE
            print(f"Frame {i+1} | Score: {max_val:.3f} | PlayerX: {abs_player_x} | BlockX_Target: {expected_block_x:.1f} | DNA_Diff: {diff:.2f}")

run_v63_forensic_heartbeat()