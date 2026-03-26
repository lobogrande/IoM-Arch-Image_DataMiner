import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json

# --- TARGET GAP CONFIG ---
# We are scouting the gap between Floor 26 and Floor 27 from your last run
GAP_START_IDX = 2194 # From your v84_perfect_map.json
GAP_END_IDX = 2497   # From your v84_perfect_map.json
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
        roi = img_gray[y1:y1+48, x1:x1+48]
        diff = min([np.sum(cv2.absdiff(roi, bg)) / 2304 for bg in bg_templates])
        vector.append(1 if diff > 7.5 else 0)
    return vector

def run_gap_forensics():
    bg_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- Scouting Gap: Frame {GAP_START_IDX} to {GAP_END_IDX} ---")
    
    # Baseline from the start of the gap
    anchor_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[GAP_START_IDX]), 0)
    active_vector = get_existence_vector(anchor_gray, bg_t)
    
    for i in range(GAP_START_IDX + 1, GAP_END_IDX):
        curr_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]), 0)
        new_vector = get_existence_vector(curr_gray, bg_t)
        
        diff_count = sum(1 for a, b in zip(active_vector, new_vector) if a != b)
        
        # We print any "events" where more than 3 bits changed
        if diff_count >= 4:
            print(f"  > Frame {i}: {diff_count} bits changed. (Potential missed floor?)")
        
        active_vector = new_vector

if __name__ == "__main__":
    run_gap_forensics()