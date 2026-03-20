import cv2
import numpy as np
import os
import re
import sys

# --- 1. THE NARROW "DIGITS-ONLY" COORDINATES ---
# Derived from your latest Pixel Probe verification
DIG_Y1, DIG_Y2 = 230, 246
DIG_X1, DIG_X2 = 250, 279

# Dataset to test
DATASET_ID = "0"
BUFFER_DIR = f"capture_buffer_{DATASET_ID}"
DIGITS_DIR = "digits"

def run_stress_test():
    print(f"--- INITIATING CONSENSUS STRESS-TEST (RUN {DATASET_ID}) ---")
    print(f"Targeting Narrow ROI: X[{DIG_X1}-{DIG_X2}] Y[{DIG_Y1}-{DIG_Y2}]")
    
    # 1. Load Digit Assets
    digit_map = load_digit_map()
    
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(('.png', '.jpg'))])
    
    print(f"Scanning {len(frames)} frames. Only non-zero/non-empty results will be logged.")
    print("-" * 50)
    
    found_count = 0
    for i, f_name in enumerate(frames):
        img = cv2.imread(os.path.join(BUFFER_DIR, f_name), 0) # Read as Grayscale
        if img is None: continue
        
        # 2. Extract Narrow ROI
        roi = img[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2]
        
        # 3. Bitwise Scan
        val = get_bitwise_number(roi, digit_map)
        
        # 4. LOGGING
        # We only print when it actually sees a number to avoid flooding the console
        if val != -1:
            found_count += 1
            print(f"[{i:05d}] {f_name} | DETECTED: {val}")
        
        if i % 500 == 0:
            sys.stdout.write(f"\r Progress: {i}/{len(frames)} frames scanned...")
            sys.stdout.flush()

    print(f"\n--- STRESS TEST COMPLETE ---")
    print(f" Total Frames: {len(frames)}")
    print(f" Successful Reads: {found_count}")
    print(f" Success Rate: {(found_count/len(frames))*100:.2f}%")

def load_digit_map():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            match = re.search(r'\d', f)
            if match:
                v = int(match.group())
                img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
                _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[v].append(b)
    return d_map

def get_bitwise_number(gray_roi, digit_map):
    # Standard high-contrast threshold
    _, bin_h = cv2.threshold(gray_roi, 195, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t in temps:
            res = cv2.matchTemplate(bin_h, t, cv2.TM_CCOEFF_NORMED)
            if res.max() >= 0.88:
                locs = np.where(res >= 0.88)
                for pt in zip(*locs[::-1]):
                    matches.append({'x': pt[0], 'val': val})
    
    if not matches: return -1
    
    matches.sort(key=lambda d: d['x'])
    unique = [m['val'] for i, m in enumerate(matches) if i == 0 or abs(m['x'] - matches[i-1]['x']) > 6]
    try:
        return int("".join(map(str, unique)))
    except:
        return -1

if __name__ == "__main__":
    run_stress_test()