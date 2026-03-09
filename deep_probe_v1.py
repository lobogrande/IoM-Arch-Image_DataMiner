import cv2
import numpy as np
import os

# --- 1. TARGET: FLOOR 12 TRANSITION ---
TARGET_FRAMES = [
    "frame_20260306_231752_273189.png", # Start of 12
    "frame_20260306_231752_317844.png",
    "frame_20260306_231752_371368.png"
]

# Verified Coordinates
HEADER_ROI = (58, 70, 105, 127) # Y1:Y2, X1:X2
DIG_ROI = (230, 246, 250, 281)
DATASET_DIR = "capture_buffer_0"
OUTPUT_DIR = "deep_probe_results"

def run_deep_probe():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print("--- INITIATING DEEP PROBE ---")

    for f_name in TARGET_FRAMES:
        img = cv2.imread(os.path.join(DATASET_DIR, f_name), 0)
        if img is None: continue

        # A. HEADER PROBE
        h_roi = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        # Export at multiple thresholds to find the "visible" one
        for t in [195, 175, 155]:
            _, b = cv2.threshold(h_roi, t, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f"{OUTPUT_DIR}/header_T{t}_{f_name}", cv2.resize(b, (0,0), fx=5, fy=5))

        # B. DIG STAGE PROBE
        d_roi = img[DIG_ROI[0]:DIG_ROI[1], DIG_ROI[2]:DIG_ROI[3]]
        for t in [195, 175, 155]:
            _, b = cv2.threshold(d_roi, t, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f"{OUTPUT_DIR}/dig_T{t}_{f_name}", cv2.resize(b, (0,0), fx=5, fy=5))

    print(f"Done. Examine '{OUTPUT_DIR}'.")
    print("Are the numbers '1' and '2' solid white? Or are they 'Swiss cheese' (full of black holes)?")

if __name__ == "__main__":
    run_deep_probe()