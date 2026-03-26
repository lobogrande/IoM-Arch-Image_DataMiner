import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import re

# --- DIAGNOSTIC CONFIG ---
TARGET_WINDOW = (650, 750) # The frames around where Floor 12 should live
HEADER_ROI = (58, 70, 105, 127)
DIGITS_DIR = cfg.DIGIT_DIR
DATASET_DIR = cfg.get_buffer_path(0)

def run_heartbeat_monitor():
    digit_map = load_digit_map_fixed()
    frames = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith(('.png', '.jpg'))])
    
    print(f"{'Frame':<40} | {'Digit':<5} | {'Conf':<6} | {'X'}")
    print("-" * 65)

    for i in range(TARGET_WINDOW[0], TARGET_WINDOW[1]):
        img = cv2.imread(os.path.join(DATASET_DIR, frames[i]), 0)
        roi = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        
        # Check at T175 (Our supposed best baseline)
        _, bin_roi = cv2.threshold(roi, 175, 255, cv2.THRESH_BINARY)
        
        for val, temps in digit_map.items():
            for t_idx, t_bin in enumerate(temps):
                res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                # We print anything above 0.50 so we can see what's almost matching
                if max_val > 0.50:
                    print(f"{frames[i]:<40} | {val:<5} | {max_val:.4f} | {max_loc[0]}")

def load_digit_map_fixed():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f)
            if m:
                v = int(m.group()); img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
                _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[v].append(b)
    return d_map

if __name__ == "__main__":
    run_heartbeat_monitor()