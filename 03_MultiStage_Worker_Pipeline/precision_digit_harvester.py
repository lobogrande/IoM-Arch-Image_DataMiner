import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- 1. TARGET FILENAMES FROM YOUR AUDIT ---
HARVEST_FILES = [
    "frame_20260306_231751_196638.png", # Stage 10 (Targeting the 1 and 0)
    "frame_20260306_231751_741608.png", # Stage 11 (Targeting the 1s)
    "frame_20260306_231753_721292.png", # Stage 14 (Targeting the ghost 4)
    "frame_20260306_231749_223602.png"  # Stage 8 (Targeting the B-shaped 8)
]

# Verified Box for the WHOLE number area
DIG_Y1, DIG_Y2, DIG_X1, DIG_X2 = 230, 246, 250, 281
DATASET_DIR = cfg.get_buffer_path(0)
DIGITS_DIR = "digits_harvest_v3" # Saving to a new folder to keep things clean

def run_blob_harvest():
    if not os.path.exists(DIGITS_DIR): os.makedirs(DIGITS_DIR)
    print(f"--- INITIATING BLOB-BASED HARVEST ---")

    for f_name in HARVEST_FILES:
        img_path = os.path.join(DATASET_DIR, f_name)
        if not os.path.exists(img_path): continue
            
        # 1. Grab the full ROI and Binarize
        img = cv2.imread(img_path, 0)
        roi = img[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2]
        _, bin_roi = cv2.threshold(roi, 195, 255, cv2.THRESH_BINARY)
        
        # 2. Find Contours (The "Blobs" of white pixels)
        contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter out tiny noise blobs (less than 4 pixels wide/tall)
            if w < 4 or h < 4: continue 
            
            digit_crop = bin_roi[y:y+h, x:x+w]
            
            # Save using the original ID to keep things organized
            out_name = f"blob_{i}_from_{f_name.split('_')[-1]}"
            cv2.imwrite(os.path.join(DIGITS_DIR, out_name), digit_crop)
            print(f" Found Digit Blob {i} in {f_name} -> {out_name} ({w}x{h})")

if __name__ == "__main__":
    run_blob_harvest()