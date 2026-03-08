import cv2
import numpy as np
import os
import json
import sys

# --- 1. CONFIGURATION ---
DATASET_ID = "0" 
BUFFER_DIR = f"capture_buffer_{DATASET_ID}"
DIGITS_DIR = "digits"
TEMPLATE_DIR = "templates"
OUTPUT_JSON = f"milestones_run_{DATASET_ID}.json"

# Anchor ROIs
HEADER_ROI = (56, 100, 16, 35)    
DIG_STAGE_ROI = (330, 185, 20, 100) 
AI_SLOTS = {i: (100 + (i % 6) * 60, 500 + (i // 6) * 65) for i in range(24)}

# --- 2. SENSOR C: OCCUPANCY (Fixed for Sensitivity) ---
def get_occupancy_fingerprint(gray_img, bg_templates):
    fingerprint = []
    for i in range(24):
        ax, ay = AI_SLOTS[i]
        slot_roi = gray_img[ay-22:ay+22, ax-22:ax+24]
        is_empty = False
        for bg_temp in bg_templates:
            res = cv2.matchTemplate(slot_roi, bg_temp, cv2.TM_CCOEFF_NORMED)
            if cv2.minMaxLoc(res)[1] > 0.82: # Slightly lowered for better detection
                is_empty = True; break
        fingerprint.append(0 if is_empty else 1)
    return fingerprint

# --- 3. EXECUTION ---
def run_sentinel_v38_1():
    print(f"--- INITIATING ADAPTIVE PASS 1 (RUN {DATASET_ID}) ---")
    
    # Load Assets
    digit_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            v = int(f[0]); img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
            if img is not None:
                _, b = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
                digit_map[v].append(b)

    bg_templates = [cv2.imread(os.path.join(TEMPLATE_DIR, f), 0) for f in os.listdir(TEMPLATE_DIR) if "background_plain" in f]
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(('.png', '.jpg'))])
    
    milestones = []
    current_brain_f = 0
    last_fingerprint = None
    
    for i, f_name in enumerate(frames):
        img_bgr = cv2.imread(os.path.join(BUFFER_DIR, f_name))
        if img_bgr is None: continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # SENSOR A & B: Independent Reading
        h_val = get_bitwise_number(gray[56:72, 100:135], digit_map)
        d_val = get_bitwise_number(gray[330:350, 185:285], digit_map)
        
        # SENSOR C: Occupancy
        curr_fingerprint = get_occupancy_fingerprint(gray, bg_templates)
        
        # ADAPTIVE TRIGGER
        # 1. Did either sensor see a floor higher than our current one?
        raw_val = max(h_val, d_val)
        
        # 2. Did the layout shift at least slightly? (Lowered to 4 for sensitivity)
        layout_shift = last_fingerprint is None or sum(abs(a - b) for a, b in zip(curr_fingerprint, last_fingerprint)) >= 4

        if raw_val > current_brain_f and layout_shift:
            milestones.append({'idx': i, 'floor': raw_val, 'frame': f_name, 'type': 'ANCHOR'})
            print(f"\n [SENTINEL] Found Floor {raw_val} at {f_name} (H:{h_val} D:{d_val})")
            
            current_brain_f = raw_val
            last_fingerprint = curr_fingerprint
            
        sys.stdout.write(f"\r Progress: {f_name} | Brain Floor: {current_brain_f} | Read: {raw_val}")
        sys.stdout.flush()

    with open(OUTPUT_JSON, 'w') as f: json.dump(milestones, f, indent=4)
    print(f"\n--- PASS 1 COMPLETE: {len(milestones)} milestones saved ---")

def get_bitwise_number(gray_roi, digit_map, min_conf=0.85):
    _, bin_h = cv2.threshold(gray_roi, 190, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t in temps:
            res = cv2.matchTemplate(bin_h, t, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]): matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = [m['val'] for i, m in enumerate(matches) if i == 0 or abs(m['x'] - matches[i-1]['x']) > 6]
    return int("".join(map(str, unique))) if unique else -1

if __name__ == "__main__":
    run_sentinel_v38_1()