import cv2
import numpy as np
import os
import re
import sys

# --- 1. YOUR VERIFIED COORDINATES ---
# Narrow ROI with your +2px horizontal buffer
DIG_Y1, DIG_Y2 = 230, 246
DIG_X1, DIG_X2 = 250, 281 # X2 increased by 2 from 279

DATASET_ID = "0"
BUFFER_DIR = f"capture_buffer_{DATASET_ID}"
DIGITS_DIR = "digits"

def run_ocr_optimizer():
    print(f"--- INITIATING OCR OPTIMIZER (RUN {DATASET_ID}) ---")
    digit_map = load_digit_map()
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(('.png', '.jpg'))])
    
    found_count = 0
    for i, f_name in enumerate(frames):
        img = cv2.imread(os.path.join(BUFFER_DIR, f_name), 0)
        if img is None: continue
        
        roi = img[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2]
        
        # MULTI-THRESHOLD SCAN
        # We try 3 levels of brightness to find the clearest digits
        best_val = -1
        for thresh in [180, 195, 210]:
            val = get_bitwise_refined(roi, digit_map, thresh)
            if val != -1:
                best_val = val
                break
        
        if best_val != -1:
            found_count += 1
            print(f"[{i:05d}] {f_name} | DETECTED: {best_val}")
        
        if i % 500 == 0:
            sys.stdout.write(f"\r Progress: {i}/{len(frames)} frames...")
            sys.stdout.flush()

    print(f"\n--- REFINEMENT COMPLETE ---")
    print(f" Total Frames: {len(frames)} | Successes: {found_count}")

def get_bitwise_refined(gray_roi, digit_map, thresh_val):
    # 1. Binarize and Denoise
    _, bin_h = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2), np.uint8)
    bin_h = cv2.morphologyEx(bin_h, cv2.MORPH_OPEN, kernel) # Removes "salt" noise

    matches = []
    for val, temps in digit_map.items():
        for t in temps:
            res = cv2.matchTemplate(bin_h, t, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.85) # Lowered confidence for blurry hits
            for pt in zip(*locs[::-1]):
                matches.append({'x': pt[0], 'val': val, 'conf': res[pt[1], pt[0]]})
    
    if not matches: return -1
    
    # 2. Conflict Resolution: Sort by X position, then highest confidence
    matches.sort(key=lambda d: (d['x'], -d['conf']))
    
    unique = []
    if matches:
        unique.append(matches[0]['val'])
        for i in range(1, len(matches)):
            # If the next digit is at least 5 pixels away, it's a new number
            if abs(matches[i]['x'] - matches[i-1]['x']) > 5:
                unique.append(matches[i]['val'])
    
    try:
        return int("".join(map(str, unique)))
    except:
        return -1

def load_digit_map():
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
    run_ocr_optimizer()