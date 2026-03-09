import cv2
import numpy as np
import os
import json
import sys
import shutil
import re
import time

# --- 1. MASTER BOSS DATA ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'},
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'},
    34: {
        'tier': 'mixed', 
        'special': {
            0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2',
            6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2',
            12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2',
            18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'
        }
    },
    35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'},
    49: {
      "tier": "mixed",
      "special": {
        0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3",
        6: "com3",  7: "com3",  8: "com3",  9: "com3",  10: "com3", 11: "com3",
        12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3",
        18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"
      }
    },
    74: {
        'tier': 'mixed', 
        'special': {
            0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3',
            6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3',
            12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3',
            18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'
        }
    },
    98: {'tier': 'myth3'},
    99: {
      "tier": "mixed",
      "special": {
        0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2",
        6: "com3",  7: "rare3",  8: "epic3",  9: "leg3",  10: "myth3", 11: "div2",
        12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2",
        18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"
      }
    }
}

# --- 2. VERIFIED COORDINATES ---
DATASETS = ["0", "1", "2", "3", "4"] 
DIGITS_DIR = "digits"
TEMPLATE_DIR = "templates"
BASE_EVIDENCE_DIR = "Pass1_Evidence"
SHOW_HUD = True

# AI ROIs (Y, X, H, W)
HEADER_ROI = (58, 105, 12, 22)      
DIG_STAGE_ROI = (233, 163, 13, 116) 
SLOT1_CENTER = (75, 261)
X_STEP, Y_STEP = 59.1, 59.1

# HUD Overlays (X1, Y1, X2, Y2)
HUD_STAGE = (103, 56, 129, 72)
HUD_DIG = (161, 231, 281, 248)

def run_calibrated_sentinel():
    start_time = time.time()
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
        # GUARANTEED START
        first_img = cv2.imread(os.path.join(buffer_path, frames[0]))
        commit_milestone(ds_id, 0, 1, frames[0], first_img, milestones, evidence_path)
        
        current_f = 1
        leap_candidate = -1; leap_counter = 0

        print(f"\n[START] Run {ds_id} | {len(frames)} frames")
        
        for i in range(1, len(frames)):
            img = cv2.imread(os.path.join(buffer_path, frames[i]))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # SENSORS
            h_val = get_bitwise_number(gray[58:70, 105:127], digit_map)
            d_val = get_bitwise_number(gray[233:246, 163:279], digit_map)
            read_val = max(h_val, d_val)
            
            if read_val > current_f:
                # 79-100 Stability Check
                # If we see a big jump or an 80/90 number, require 2 frames of proof
                if (read_val - current_f) > 10 or 79 <= read_val <= 100:
                    if read_val == leap_candidate:
                        leap_counter += 1
                    else:
                        leap_candidate = read_val
                        leap_counter = 1
                    
                    if leap_counter >= 2: # High-confidence trigger
                        commit_milestone(ds_id, i, read_val, frames[i], img, milestones, evidence_path)
                        current_f = read_val
                        leap_candidate = -1; leap_counter = 0
                else:
                    commit_milestone(ds_id, i, read_val, frames[i], img, milestones, evidence_path)
                    current_f = read_val
                    leap_candidate = -1; leap_counter = 0
            
            if i % 100 == 0:
                sys.stdout.write(f"\r Run {ds_id} | {frames[i]} | Floor: {current_f} | Read: {read_val}")
                sys.stdout.flush()

        with open(f"milestones_run_{ds_id}.json", 'w') as f:
            json.dump(milestones, f, indent=4)
        perform_gap_audit(ds_id, milestones, current_f)

    print(f"\n\n--- ALL DATASETS PROCESSED IN {time.time()-start_time:.2f}s ---")

def commit_milestone(ds_id, idx, floor, f_name, img, milestones, evidence_path):
    milestones.append({'idx': idx, 'floor': floor, 'frame': f_name})
    out_file = f"{evidence_path}/F{floor}_{f_name}"
    
    if SHOW_HUD:
        marked = img.copy()
        cv2.rectangle(marked, (HUD_STAGE[0], HUD_STAGE[1]), (HUD_STAGE[2], HUD_STAGE[3]), (255, 0, 255), 1)
        cv2.rectangle(marked, (HUD_DIG[0], HUD_DIG[1]), (HUD_DIG[2], HUD_DIG[3]), (255, 0, 255), 1)
        
        if floor in BOSS_DATA:
            for i in range(24):
                row, col = divmod(i, 6)
                cx = int(SLOT1_CENTER[0] + (col * X_STEP))
                cy = int(SLOT1_CENTER[1] + (row * Y_STEP))
                cv2.rectangle(marked, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
                tier = BOSS_DATA[floor]['special'].get(i, BOSS_DATA[floor]['tier']) if 'special' in BOSS_DATA[floor] else BOSS_DATA[floor]['tier']
                cv2.putText(marked, tier[:5], (cx-22, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imwrite(out_file, marked)
    else:
        cv2.imwrite(out_file, img)
    print(f"\n [RUN {ds_id}] Anchor: Floor {floor}")

def load_digit_map_robust():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f)
            if m:
                v = int(m.group()); img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
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
    print(f" Missing: {len(missing)} floors: {missing[:15]}...")

if __name__ == "__main__":
    run_calibrated_sentinel()