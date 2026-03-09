import cv2
import numpy as np
import os
import json
import sys
import shutil
import re
import time

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
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

# --- 2. VERIFIED CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
DIGITS_DIR = "digits"
BASE_EVIDENCE_DIR = "Pass1_Evidence"
SHOW_HUD = True

# ROIs for Scanning (Y, X, H, W)
HEADER_Y1, HEADER_Y2, HEADER_X1, HEADER_X2 = 58, 70, 105, 127
DIG_Y1, DIG_Y2, DIG_X1, DIG_X2 = 230, 246, 250, 281

# HUD Drawing Coordinates (X1, Y1, X2, Y2)
HUD_STAGE = (103, 56, 129, 72)
HUD_DIG_WIDE = (161, 231, 281, 248)
HUD_DIG_NARROW = (250, 230, 281, 246)
SLOT1_CENTER = (75, 261)
X_STEP, Y_STEP = 59.1, 59.1

def run_adaptive_sentinel():
    start_time = time.time()
    # PRE-PROCESS TEMPLATES ONCE AT HIGH CONTRAST
    digit_map = load_digit_map_fixed()
    
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
        # Initial Anchor
        first_img = cv2.imread(os.path.join(buffer_path, frames[0]))
        commit_milestone(ds_id, 0, 1, frames[0], first_img, milestones, evidence_path)
        
        current_f = 1
        stab_candidate, stab_counter = -1, 0
        perf_timer = time.time()

        print(f"\n[START] Run {ds_id} | {len(frames)} frames")
        
        for i in range(1, len(frames)):
            img = cv2.imread(os.path.join(buffer_path, frames[i]))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # CONSENSUS SENSORS
            h_val = -1
            for t in [195, 175]:
                h_val = get_bitwise_adaptive(gray[HEADER_Y1:HEADER_Y2, HEADER_X1:HEADER_X2], digit_map, t, 0.82)
                if h_val != -1: break
                
            d_val = -1
            for t in [155, 165, 175]: # Focus on lower thresholds where transitions are visible
                d_val = get_bitwise_adaptive(gray[DIG_Y1:DIG_Y2, DIG_X1:DIG_X2], digit_map, t, 0.72)
                if d_val != -1: break
            
            # CORE LOGIC: Trust matching sensors or a stable Header
            raw_read = -1
            if h_val > current_f and (h_val == d_val or d_val == -1):
                raw_read = h_val
            elif d_val > current_f and d_val == h_val:
                raw_read = d_val
                
            if raw_read != -1:
                if raw_read == stab_candidate: stab_counter += 1
                else: stab_candidate, stab_counter = raw_read, 1
                
                if stab_counter >= 2: # Success: static confirmed transition
                    current_f = stab_candidate
                    commit_milestone(ds_id, i, current_f, frames[i], img, milestones, evidence_path)
                    stab_candidate, stab_counter = -1, 0
            else:
                stab_candidate, stab_counter = -1, 0

            if i % 100 == 0:
                fps = 100 / (time.time() - perf_timer)
                sys.stdout.write(f"\r Run {ds_id} | {frames[i]} | Floor: {current_f} | H:{h_val} D:{d_val} | FPS: {fps:.1f}")
                sys.stdout.flush()
                perf_timer = time.time()

        with open(f"milestones_run_{ds_id}.json", 'w') as f:
            json.dump(milestones, f, indent=4)
        perform_audit(ds_id, milestones, current_f)

def get_bitwise_adaptive(gray_roi, digit_map, thresh_val, min_conf):
    _, bin_roi = cv2.threshold(gray_roi, thresh_val, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t_bin in temps:
            # Match current ROI against the pre-thresholded templates
            res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]): matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = [m['val'] for i, m in enumerate(matches) if i == 0 or abs(m['x'] - matches[i-1]['x']) > 6]
    try:
        return int("".join(map(str, unique))) if unique else -1
    except:
        return -1

def load_digit_map_fixed():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f)
            if m:
                v = int(m.group()); img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
                # Lock templates at high-contrast 195
                _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[v].append(b)
    return d_map

def commit_milestone(ds_id, idx, floor, f_name, img, milestones, evidence_path):
    milestones.append({'idx': idx, 'floor': floor, 'frame': f_name})
    out_file = f"{evidence_path}/F{floor}_{f_name}"
    marked = img.copy()
    cv2.rectangle(marked, (HUD_STAGE[0], HUD_STAGE[1]), (HUD_STAGE[2], HUD_STAGE[3]), (255, 0, 255), 1)
    cv2.rectangle(marked, (HUD_DIG_WIDE[0], HUD_DIG_WIDE[1]), (HUD_DIG_WIDE[2], HUD_DIG_WIDE[3]), (255, 0, 255), 1)
    cv2.rectangle(marked, (HUD_DIG_NARROW[0], HUD_DIG_NARROW[1]), (HUD_DIG_NARROW[2], HUD_DIG_NARROW[3]), (0, 255, 0), 1)
    
    if floor in BOSS_DATA:
        for b_idx in range(24):
            row, col = divmod(b_idx, 6)
            cx, cy = int(75 + (col * 59.1)), int(261 + (row * 59.1))
            cv2.rectangle(marked, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
            tier = BOSS_DATA[floor]['special'].get(b_idx, BOSS_DATA[floor]['tier']) if 'special' in BOSS_DATA[floor] else BOSS_DATA[floor]['tier']
            cv2.putText(marked, tier[:5], (cx-22, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imwrite(out_file, marked)
    print(f"\n [RUN {ds_id}] Anchor Saved: Floor {floor}")

def perform_audit(run_id, milestones, max_floor):
    found = set(m['floor'] for m in milestones)
    missing = sorted(list(set(range(1, max_floor + 1)) - found))
    print(f"\n--- GAP AUDIT RUN {run_id} ---")
    print(f" Found: {len(milestones)} | Missing: {len(missing)} floors.")

if __name__ == "__main__":
    run_adaptive_sentinel()