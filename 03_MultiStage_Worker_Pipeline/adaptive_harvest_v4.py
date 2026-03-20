import cv2
import numpy as np
import os

# --- 1. TARGET FRAME (Stage 14) ---
FRAME_PATH = "capture_buffer_0/frame_20260306_231753_721292.png"
DIG_Y1, DIG_Y2, DIG_X1, DIG_X2 = 230, 246, 250, 281
OUTPUT_DIR = "harvest_v4_adaptive"

def run_adaptive_harvest():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    img = cv2.imread(FRAME_PATH, 0)
    roi = img[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2]

    print(f"--- ATTEMPTING ADAPTIVE HARVEST ON STAGE 14 ---")
    
    # We test lower thresholds to 'reconstruct' the eroded '4'
    for thresh in [185, 175, 165]:
        _, bin_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        print(f" Threshold {thresh}: Found {len(contours)} blobs.")
        
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 3 or h < 3: continue # Filter speckles
            
            digit_crop = bin_roi[y:y+h, x:x+w]
            out_name = f"thresh{thresh}_blob{i}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), digit_crop)

if __name__ == "__main__":
    run_adaptive_harvest()