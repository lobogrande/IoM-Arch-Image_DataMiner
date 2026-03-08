import cv2
import numpy as np
import os
import json
import sys
import shutil
import re
import time

# --- 1. THE GROUND TRUTH ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'},
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'},
    34: {'tier': 'mixed', 'special': {8: 'myth1', 9: 'myth1', 14: 'myth1', 15: 'myth1'}},
    49: {'tier': 'mixed', 'special': {0: 'dirt3', 18: 'myth2'}},
    74: {'tier': 'mixed', 'special': {20: 'div1', 21: 'div1'}},
    98: {'tier': 'myth3'}, 99: {'tier': 'mixed'}
}

# --- 2. CONFIGURATION ---
DATASETS = ["0", "1", "2", "3", "4"] 
DIGITS_DIR = "digits"
TEMPLATE_DIR = "templates"
BASE_EVIDENCE_DIR = "Pass1_Evidence"
SHOW_HUD = True 

# Analysis & HUD Coordinates
HUD_COORDS = {i: (73 + (i % 6) * 59, 268 + (i // 6) * 61) for i in range(24)}
HUD_BOX_TL = (185, 330) 
HUD_BOX_BR = (285, 350) 

def run_observational_sentinel():
    start_time = time.time()
    total_frames = 0
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
        total_frames += len(frames)

        milestones = []
        first_img = cv2.imread(os.path.join(buffer_path, frames[0]))
        commit_milestone(ds_id, 0, 1, frames[0], first_img, milestones, evidence_path)
        
        current_f = 1
        leap_candidate = -1; leap_counter = 0

        print(f"\n[START] Run {ds_id} | Dataset: {len(frames)} frames")
        
        for i in range(1, len(frames)):
            img = cv2.imread(os.path.join(buffer_path, frames[i]))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Numeric Sensors
            h_val = get_bitwise_number(gray[56:72, 100:135], digit_map)
            d_val = get_bitwise_number(gray[330:350, 185:285], digit_map)
            read_val = max(h_val, d_val)
            
            if read_val > current_f:
                if (read_val - current_f) > 10:
                    if read_val == leap_candidate: leap_counter += 1
                    else: leap_candidate = read_val; leap_counter = 1
                    
                    if leap_counter >= 5: # Stable leap
                        commit_milestone(ds_id, i, read_val, frames[i], img, milestones, evidence_path)
                        current_f = read_val
                        leap_candidate = -1; leap_counter = 0
                else:
                    commit_milestone(ds_id, i, read_val, frames[i], img, milestones, evidence_path)
                    current_f = read_val
                    leap_candidate = -1; leap_counter = 0
            
            if i % 100 == 0:
                sys.stdout.write(f"\r Run {ds_id} | {frames[i]} | Max Floor: {current_f} | Read: {read_val}")
                sys.stdout.flush()

        with open(f"milestones_run_{ds_id}.json", 'w') as f:
            json.dump(milestones, f, indent=4)
        perform_gap_audit(ds_id, milestones, current_f)

    # FINAL PERFORMANCE REPORT
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n\n--- FINAL PERFORMANCE REPORT ---")
    print(f" Total Datasets Processed: {len(DATASETS)}")
    print(f" Total Frames Scanned: {total_frames}")
    print(f" Total Time: {elapsed:.2f}s")
    print(f" Average Processing Speed: {total_frames / elapsed:.2f} FPS")
    print(f" Evidence Library: {BASE_EVIDENCE_DIR}")

def commit_milestone(ds_id, idx, floor, f_name, img, milestones, evidence_path):
    milestones.append({'idx': idx, 'floor': floor, 'frame': f_name})
    out_file = f"{evidence_path}/F{floor}_{f_name}"
    
    if SHOW_HUD:
        marked = img.copy()
        # Transition Box
        cv2.rectangle(marked, HUD_BOX_TL, HUD_BOX_BR, (255, 0, 255), 2)
        
        # SOFT BOSS CHECK: Overlay layout if floor is a known boss
        if floor in BOSS_DATA:
            for i in range(24):
                hx, hy = HUD_COORDS[i]
                cv2.rectangle(marked, (hx-24, hy-24), (hx+24, hy+24), (0, 255, 0), 1)
                tier_label = BOSS_DATA[floor]['special'].get(i, BOSS_DATA[floor]['tier']) if 'special' in BOSS_DATA[floor] else BOSS_DATA[floor]['tier']
                cv2.putText(marked, tier_label[:5], (hx-22, hy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        cv2.imwrite(out_file, marked)
    else:
        cv2.imwrite(out_file, img)
    print(f"\n [RUN {ds_id}] Milestone Saved: Floor {floor}")

def load_digit_map_robust():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f)
            if m:
                v = int(m.group())
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
    print(f" GAP AUDIT: Run {run_id} missing {len(missing)} floors: {missing[:10]}...")

if __name__ == "__main__":
    run_observational_sentinel()