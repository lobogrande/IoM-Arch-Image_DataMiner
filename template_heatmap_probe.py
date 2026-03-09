import cv2
import numpy as np
import os
import re

# --- 1. SETUP ---
TEST_FRAME = "capture_buffer_0/frame_20260306_231753_721292.png" # Stage 14
DIGITS_DIR = "digits"
ROI = (230, 250, 16, 31) # Y, X, H, W (Your verified AI box)

def run_heatmap_probe():
    img = cv2.imread(TEST_FRAME, 0)
    if img is None: return
    roi = img[ROI[0]:ROI[0]+ROI[2], ROI[1]:ROI[1]+ROI[3]]
    _, bin_roi = cv2.threshold(roi, 185, 255, cv2.THRESH_BINARY) # Lowered for probe

    print(f"--- CONFIDENCE AUDIT FOR {TEST_FRAME} ---")
    
    for f in sorted(os.listdir(DIGITS_DIR)):
        if not f.endswith('.png'): continue
        temp = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
        _, bin_temp = cv2.threshold(temp, 195, 255, cv2.THRESH_BINARY)
        
        # Run template match
        res = cv2.matchTemplate(bin_roi, bin_temp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Filter for the relevant digits (1 and 4 for this frame)
        if f.startswith(('1', '4')):
            print(f" Template: {f:20} | Max Confidence: {max_val:.4f}")

if __name__ == "__main__":
    run_heatmap_probe()