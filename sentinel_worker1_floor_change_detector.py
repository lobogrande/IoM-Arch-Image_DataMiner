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
ANCHOR_FILE = "dig_stage_anchor.png"

# PADDED Scanning ROIs (Y1, Y2, X1, X2)
HEADER_ROI = (54, 74, 103, 138)
DIG_VAL_ROI = (230, 246, 250, 281)
DIG_ANCH_ROI = (229, 248, 163, 253)

def run_resilient_sentinel():
    start_time = time.time()
    digit_map = load_digit_map_fixed()
    anchor_tmpl = cv2.imread(ANCHOR_FILE, 0) if os.path.exists(ANCHOR_FILE) else None
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
        h_candidate, h_count = -1, 0
        perf_timer = time.time()

        print(f"\n[START] Run {ds_id} | {len(frames)} frames")
        
        for i in range(1, len(frames)):
            img = cv2.imread(os.path.join(buffer_path, frames[i]))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # SENSOR 1: Header
            h_val = get_bitwise_precision(gray[54:74, 103:138], digit_map, 175, 0.82)
            
            # SENSOR 2: Dig Stage + Anchor
            has_anchor = False
            if anchor_tmpl is not None:
                res = cv2.matchTemplate(gray[229:248, 163:253], anchor_tmpl, cv2.TM_CCOEFF_NORMED)
                if res.max() > 0.60: has_anchor = True

            d_val = get_bitwise_precision(gray[230:246, 250:281], digit_map, 165, 0.72)

            # --- RESILIENT LOGIC BRIDGE ---
            if h_val > current_f:
                jump_size = h_val - current_f
                
                # CONDITION A: Large Jumps (>15) require Triple-Consensus
                if jump_size > 15:
                    if h_val == d_val and has_anchor:
                        current_f = h_val
                        commit_milestone(ds_id, i, current_f, frames[i], img, milestones, evidence_path)
                        h_candidate, h_count = -1, 0
                
                # CONDITION B: Standard Progression uses 3-frame stability
                else:
                    if h_val == h_candidate: h_count += 1
                    else: h_candidate, h_count = h_val, 1
                    
                    if h_count >= 3:
                        current_f = h_val
                        commit_milestone(ds_id, i, current_f, frames[i], img, milestones, evidence_path)
                        h_candidate, h_count = -1, 0
            else:
                h_candidate, h_count = -1, 0

            if i % 100 == 0:
                fps = 100 / (time.time() - perf_timer)
                sys.stdout.write(f"\r Run {ds_id} | {frames[i]} | Floor: {current_f} | H:{h_val} D:{d_val} | FPS: {fps:.1f}")
                sys.stdout.flush()
                perf_timer = time.time()

        with open(f"milestones_run_{ds_id}.json", 'w') as f:
            json.dump(milestones, f, indent=4)
        perform_gap_audit_detailed(ds_id, milestones, current_f)

def get_bitwise_precision(roi, digit_map, thresh, min_conf):
    _, bin_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t_bin in temps:
            res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= min_conf)
            for pt in zip(*locs[::-1]): matches.append({'x': pt[0], 'val': val})
    if not matches: return -1
    matches.sort(key=lambda d: d['x'])
    unique = []
    for m in matches:
        if not unique or abs(m['x'] - last_x) > 4:
            unique.append(m['val']); last_x = m['x']
    try: return int("".join(map(str, unique)))
    except: return -1

def load_digit_map_fixed():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f); v = int(m.group()) if m else None
            if v is not None:
                img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
                _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[v].append(b)
    return d_map

def commit_milestone(ds_id, idx, floor, f_name, img, milestones, evidence_path):
    milestones.append({'idx': idx, 'floor': floor, 'frame': f_name})
    out_file = f"{evidence_path}/F{floor}_{f_name}"
    marked = img.copy()
    cv2.rectangle(marked, (103, 54), (138, 74), (255, 0, 255), 1)
    cv2.rectangle(marked, (161, 230), (281, 246), (255, 0, 255), 1)
    if floor in BOSS_DATA:
        for b_idx in range(24):
            row, col = divmod(b_idx, 6)
            cx, cy = int(75 + (col * 59.1)), int(261 + (row * 59.1))
            cv2.rectangle(marked, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
            tier = BOSS_DATA[floor]['special'].get(b_idx, BOSS_DATA[floor]['tier']) if 'special' in BOSS_DATA[floor] else BOSS_DATA[floor]['tier']
            cv2.putText(marked, tier[:5], (cx-22, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.imwrite(out_file, marked)

def perform_gap_audit_detailed(run_id, milestones, max_floor):
    found = sorted(list(set(m['floor'] for m in milestones)))
    missing = sorted(list(set(range(1, max_floor + 1)) - set(found)))
    print(f"\n--- GAP AUDIT RUN {run_id} ---")
    print(f" Found: {len(found)} | Missing: {len(missing)} floors")
    if missing:
        ranges = []
        start = missing[0]
        for i in range(1, len(missing)):
            if missing[i] != missing[i-1] + 1:
                ranges.append(f"{start}-{missing[i-1]}" if start != missing[i-1] else f"{start}")
                start = missing[i]
        ranges.append(f"{start}-{missing[-1]}" if start != missing[-1] else f"{start}")
        print(f" Missing Ranges: {', '.join(ranges)}")

if __name__ == "__main__":
    run_resilient_sentinel()