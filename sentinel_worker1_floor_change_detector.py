import cv2
import numpy as np
import os
import json
import sys
import shutil
import re

# --- 1. CONFIGURATION ---
DATASETS = ["0", "1", "2", "3", "4"] 
DIGITS_DIR = "digits"
TEMPLATE_DIR = "templates"
BASE_EVIDENCE_DIR = "Pass1_Evidence"
SHOW_HUD = True # Set to False for pure, unmarked images

# Analysis ROIs (Y, X, H, W)
HEADER_ROI = (56, 100, 16, 35)    
DIG_STAGE_ROI = (330, 185, 20, 100) 

# HUD Overlay Coordinates (Using your corrected X,Y)
# Aligned to the top of the grid where the "Dig Stage" text appears
HUD_BOX_TL = (185, 330) 
HUD_BOX_BR = (285, 350) 

def run_forensic_sentinel():
    print(f"--- INITIATING v38.6 FORENSIC SENTINEL ---")
    digit_map = load_digit_map_robust()
    
    if not os.path.exists(BASE_EVIDENCE_DIR): os.makedirs(BASE_EVIDENCE_DIR)

    for ds_id in DATASETS:
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.isdir(buffer_path): continue
        
        evidence_path = os.path.join(BASE_EVIDENCE_DIR, f"Run_{ds_id}")
        if os.path.exists(evidence_path): shutil.rmtree(evidence_path)
        os.makedirs(evidence_path)
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        if not frames: continue

        milestones = []
        # GUARANTEED START: Your datasets always begin with Floor 1
        first_img = cv2.imread(os.path.join(buffer_path, frames[0]))
        commit_milestone(ds_id, 0, 1, frames[0], first_img, milestones, evidence_path)
        
        current_f = 1
        leap_candidate = -1; leap_counter = 0

        print(f"\n[START] Run {ds_id} | Dataset size: {len(frames)} frames")
        
        for i in range(1, len(frames)):
            f_name = frames[i]
            img = cv2.imread(os.path.join(buffer_path, f_name))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # SENSORS
            h_val = get_bitwise_number(gray[56:72, 100:135], digit_map)
            d_val = get_bitwise_number(gray[330:350, 185:285], digit_map)
            read_val = max(h_val, d_val)
            
            # ADAPTIVE LEAP LOGIC
            if read_val > current_f:
                if (read_val - current_f) > 10:
                    if read_val == leap_candidate: leap_counter += 1
                    else: leap_candidate = read_val; leap_counter = 1
                    
                    if leap_counter >= 5: # Confirm leap after 5-frame stability
                        commit_milestone(ds_id, i, read_val, f_name, img, milestones, evidence_path)
                        current_f = read_val
                        leap_candidate = -1; leap_counter = 0
                else:
                    commit_milestone(ds_id, i, read_val, f_name, img, milestones, evidence_path)
                    current_f = read_val
                    leap_candidate = -1; leap_counter = 0
            
            if i % 100 == 0:
                sys.stdout.write(f"\r Run {ds_id} | {f_name} | Max Floor: {current_f} | Read: {read_val}")
                sys.stdout.flush()

        with open(f"milestones_run_{ds_id}.json", 'w') as f:
            json.dump(milestones, f, indent=4)
        perform_gap_audit(ds_id, milestones, current_f)

def commit_milestone(ds_id, idx, floor, f_name, img, milestones, evidence_path):
    milestones.append({'idx': idx, 'floor': floor, 'frame': f_name})
    out_file = f"{evidence_path}/F{floor}_{f_name}"
    
    if SHOW_HUD:
        marked = img.copy()
        # Purple box drawn in the specific HUD location at top of grid
        cv2.rectangle(marked, HUD_BOX_TL, HUD_BOX_BR, (255, 0, 255), 2)
        cv2.imwrite(out_file, marked)
    else:
        cv2.imwrite(out_file, img)
    
    print(f"\n [RUN {ds_id}] Milestone: Floor {floor}")

def load_digit_map_robust():
    """Extracts the first numeric character from file names (e.g., '1_bright_0.png' or '7.png')"""
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

def get_bitwise_number(gray_roi, digit_map, min_conf=0.88):
    _, bin_h = cv2.threshold(gray_roi, 195, 255, cv2.THRESH_BINARY)
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

def perform_gap_audit(run_id, milestones, max_floor):
    found = set(m['floor'] for m in milestones)
    missing = sorted(list(set(range(1, max_floor + 1)) - found))
    print(f"\n--- GAP AUDIT RUN {run_id} ---")
    print(f" Found: {len(milestones)} | Missing: {missing}")

if __name__ == "__main__":
    run_forensic_sentinel()