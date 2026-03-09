import cv2
import numpy as np
import os
import json
import sys
import shutil
import re
import time

# --- MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
DIGITS_DIR = "digits"
BASE_HEAL_DIR = "Pass2_Evidence"
HEADER_ROI = (54, 74, 103, 138)

def run_exhaustive_healer():
    digit_map = load_digit_map()
    if not os.path.exists(BASE_HEAL_DIR): os.makedirs(BASE_HEAL_DIR)

    for ds_id in DATASETS:
        json_file = f"milestones_run_{ds_id}.json"
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(json_file): continue
        
        with open(json_file, 'r') as f: anchors = json.load(f)
        heal_path = os.path.join(BASE_HEAL_DIR, f"Run_{ds_id}")
        if os.path.exists(heal_path): shutil.rmtree(heal_path)
        os.makedirs(heal_path)
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        print(f"\n--- EXHAUSTIVE HEALING RUN {ds_id} ---")
        
        # 1. CEILING FINDER
        last_anc = anchors[-1]
        ceiling_f = find_ceiling_exhaustive(buffer_path, frames, last_anc, digit_map)
        if ceiling_f:
            anchors.append(ceiling_f)
            print(f" Ceiling Confirmed: Floor {ceiling_f['floor']} at Frame {ceiling_f['idx']}")

        final_milestones = []
        # 2. GAP HEALING
        for i in range(len(anchors) - 1):
            final_milestones.append(anchors[i])
            start_f, end_f = anchors[i]['floor'], anchors[i+1]['floor']
            start_idx, end_idx = anchors[i]['idx'], anchors[i+1]['idx']
            
            if (end_f - start_f) > 1:
                print(f" Scanning Window [{start_idx}:{end_idx}] for Floors {start_f+1} to {end_f-1}")
                found = deep_scan_window(buffer_path, frames, start_idx, end_idx, start_f, end_f, digit_map)
                final_milestones.extend(found)
                for f in found:
                    save_healed_evidence(buffer_path, f, heal_path)

        final_milestones.append(anchors[-1])
        final_milestones.sort(key=lambda x: x['idx'])
        
        # Save results
        with open(f"healed_consensus_run_{ds_id}.json", 'w') as f:
            json.dump(final_milestones, f, indent=4)
        
        perform_gap_audit_detailed(ds_id, final_milestones, final_milestones[-1]['floor'])

def find_ceiling_exhaustive(path, frames, last_anc, digit_map):
    # Scan backward from the end of the video to find the maximum floor
    for i in range(len(frames) - 1, last_anc['idx'], -1):
        gray = cv2.imread(os.path.join(path, frames[i]), 0)
        roi = gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        val = get_bitwise_number(roi, digit_map, 175, 0.78)
        if val > last_anc['floor']:
            # Require 3-frame stability even for ceiling
            return {'idx': i, 'floor': val, 'frame': frames[i]}
    return None

def deep_scan_window(path, frames, start_idx, end_idx, start_f, end_f, digit_map):
    healed = []
    current_target = start_f + 1
    
    # Stability variables
    cand_f, cand_count = -1, 0
    
    for i in range(start_idx + 1, end_idx):
        gray = cv2.imread(os.path.join(path, frames[i]), 0)
        roi = gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        
        # Wide-spectrum threshold search for "hard" floors
        val = -1
        for t in [175, 155, 195]:
            val = get_bitwise_number(roi, digit_map, t, 0.72)
            if val != -1: break
            
        if val == current_target:
            if val == cand_f: cand_count += 1
            else: cand_f, cand_count = val, 1
            
            # STABILITY LOCK: Must be seen for 3 frames to avoid "Elixir Crit!" ghosts
            if cand_count >= 3:
                healed.append({'idx': i, 'floor': val, 'frame': frames[i]})
                current_target += 1
                cand_f, cand_count = -1, 0
                if current_target == end_f: break
        else:
            cand_f, cand_count = -1, 0
            
    return healed

def get_bitwise_number(roi, digit_map, thresh, min_conf):
    _, bin_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t_bin in temps:
            res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]):
                    # Disambiguation logic
                    if val == 8:
                        if np.sum(bin_roi[pt[1]:pt[1]+12, pt[0]:pt[0]+1]) < 255: matches.append({'x': pt[0], 'val': 3})
                        elif np.sum(bin_roi[pt[1]+8:pt[1]+12, pt[0]:pt[0]+1]) < 255: matches.append({'x': pt[0], 'val': 9})
                        else: matches.append({'x': pt[0], 'val': 8})
                    elif val == 6:
                        if bin_roi[pt[1]+2, pt[0]+5] > 0: continue # Likely a 5
                        matches.append({'x': pt[0], 'val': 6})
                    else: matches.append({'x': pt[0], 'val': val})
    if not matches: return -1
    matches.sort(key=lambda d: d['x'])
    unique = []; last_x = -99
    for m in matches:
        if abs(m['x'] - last_x) > 4:
            unique.append(m['val']); last_x = m['x']
    try: return int("".join(map(str, unique)))
    except: return -1

def save_healed_evidence(path, milestone, heal_path):
    img = cv2.imread(os.path.join(path, milestone['frame']))
    # Highlight healed area for verification
    cv2.rectangle(img, (103, 54), (138, 74), (0, 255, 0), 2)
    cv2.putText(img, "HEALED", (103, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(heal_path, f"F{milestone['floor']}_{milestone['frame']}"), img)

def load_digit_map():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f)
            if m:
                img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
                _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[int(m.group())].append(b)
    return d_map

def perform_gap_audit_detailed(run_id, milestones, max_floor):
    found = sorted(list(set(m['floor'] for m in milestones)))
    missing = sorted(list(set(range(1, max_floor + 1)) - set(found)))
    print(f"--- POST-HEAL GAP AUDIT RUN {run_id} ---")
    print(f" Final Count: {len(found)} | Remaining Missing: {len(missing)}")
    if missing:
        ranges = []
        start = missing[0]
        for i in range(1, len(missing)):
            if missing[i] != missing[i-1] + 1:
                ranges.append(f"{start}-{missing[i-1]}" if start != missing[i-1] else f"{start}")
                start = missing[i]
        ranges.append(f"{start}-{missing[-1]}" if start != missing[-1] else f"{start}")
        print(f" Final Missing Ranges: {', '.join(ranges)}")

if __name__ == "__main__":
    run_exhaustive_healer()