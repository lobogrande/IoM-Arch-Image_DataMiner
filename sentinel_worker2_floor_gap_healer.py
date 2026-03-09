import cv2
import numpy as np
import os
import json
import shutil
import re
import time

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
DIGITS_DIR = "digits"
BASE_HEAL_DIR = "Pass2_Evidence"
HEADER_ROI = (52, 76, 100, 142)
GRID_ROI = (250, 500, 75, 425) # ROI for board-change detection

def run_infallible_healer():
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
        print(f"\n--- INFALLIBLE HEALING RUN {ds_id} ---")
        
        # 1. CEILING DISCOVERY
        ceiling_f = find_ceiling_verified(buffer_path, frames, anchors[-1], digit_map)
        if ceiling_f: anchors.append(ceiling_f)

        final_milestones = []
        # 2. THE SIEGE: Fill every single gap
        for i in range(len(anchors) - 1):
            final_milestones.append(anchors[i])
            s_f, e_f = anchors[i]['floor'], anchors[i+1]['floor']
            s_idx, e_idx = anchors[i]['idx'], anchors[i+1]['idx']
            
            if (e_f - s_f) > 1:
                print(f" Siege Triggered: F{s_f}->F{e_f} (Need {e_f-s_f-1})")
                healed = solve_gap_recursively(buffer_path, frames, s_idx, e_idx, s_f, e_f, digit_map)
                final_milestones.extend(healed)
                for h in healed: save_healed_evidence(buffer_path, h, heal_path)

        final_milestones.append(anchors[-1])
        final_milestones.sort(key=lambda x: x['idx'])
        
        with open(f"healed_consensus_run_{ds_id}.json", 'w') as f:
            json.dump(final_milestones, f, indent=4)
        perform_final_audit(ds_id, final_milestones)

def solve_gap_recursively(path, frames, start_idx, end_idx, start_f, end_f, digit_map):
    healed = []
    missing_floors = list(range(start_f + 1, end_f))
    current_min = start_idx + 1
    
    for target in missing_floors:
        found_m = None
        # PASS 1: High-Speed Skip Scan
        for i in range(current_min, end_idx, 5):
            gray = cv2.imread(os.path.join(path, frames[i]), 0)
            roi = gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            if get_bitwise_verified(roi, digit_map, 175, 0.70) == target:
                found_m = {'idx': i, 'floor': target, 'frame': frames[i]}
                break
        
        # PASS 2: Recursive Deep Siege (Every Frame + Multi-Thresh)
        if not found_m:
            for t in [155, 195, 145, 205]:
                for i in range(current_min, end_idx):
                    gray = cv2.imread(os.path.join(path, frames[i]), 0)
                    roi = gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                    if get_bitwise_verified(roi, digit_map, t, 0.65) == target:
                        found_m = {'idx': i, 'floor': target, 'frame': frames[i]}
                        break
                if found_m: break

        # PASS 3: Blind Heuristic (Transition Detection)
        if not found_m:
            print(f" !! OCR Hard-Fail F{target}. Nominating via Transition Heuristic...")
            found_m = nominate_by_transition(path, frames, current_min, end_idx, target)

        healed.append(found_m)
        current_min = found_m['idx'] + 1
        
    return healed

def nominate_by_transition(path, frames, start, end, floor):
    # Find frame with largest pixel difference (board clear)
    max_diff = -1
    best_idx = start
    prev_img = None
    
    for i in range(start, end):
        img = cv2.imread(os.path.join(path, frames[i]), 0)
        grid = img[GRID_ROI[0]:GRID_ROI[1], GRID_ROI[2]:GRID_ROI[3]]
        if prev_img is not None:
            diff = np.sum(cv2.absdiff(grid, prev_img))
            if diff > max_diff:
                max_diff = diff
                best_idx = i
        prev_img = grid
    return {'idx': best_idx, 'floor': floor, 'frame': frames[best_idx]}

def find_ceiling_verified(path, frames, last_anc, d_map):
    for i in range(len(frames) - 1, last_anc['idx'], -5):
        gray = cv2.imread(os.path.join(path, frames[i]), 0)
        val = get_bitwise_verified(gray[52:76, 100:142], d_map, 165, 0.68)
        if val > last_anc['floor']: return {'idx': i, 'floor': val, 'frame': frames[i]}
    return None

def get_bitwise_verified(roi, digit_map, thresh, min_conf):
    _, bin_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t_bin in temps:
            res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]):
                    # Geometric Verification
                    if val == 6 and bin_roi[pt[1]+2, pt[0]+5] > 0: continue
                    if val == 8:
                        if np.sum(bin_roi[pt[1]:pt[1]+12, pt[0]:pt[0]+1]) < 255: matches.append({'x': pt[0], 'val': 3})
                        elif np.sum(bin_roi[pt[1]+8:pt[1]+12, pt[0]:pt[0]+1]) < 255: matches.append({'x': pt[0], 'val': 9})
                        else: matches.append({'x': pt[0], 'val': 8})
                    else: matches.append({'x': pt[0], 'val': val})
    if not matches: return -1
    matches.sort(key=lambda d: d['x'])
    unique = []; lx = -99
    for m in matches:
        if abs(m['x'] - lx) > 4: unique.append(m['val']); lx = m['x']
    try: return int("".join(map(str, unique)))
    except: return -1

def save_healed_evidence(path, milestone, heal_path):
    img = cv2.imread(os.path.join(path, milestone['frame']))
    cv2.rectangle(img, (100, 52), (142, 76), (0, 0, 255), 2)
    cv2.putText(img, f"HEALED F{milestone['floor']}", (100, 48), 0, 0.4, (0, 0, 255), 1)
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

def perform_final_audit(run_id, milestones):
    found = sorted(list(set(m['floor'] for m in milestones)))
    max_f = milestones[-1]['floor']
    missing = sorted(list(set(range(1, max_f + 1)) - set(found)))
    print(f"--- FINAL AUDIT RUN {run_id} ---")
    print(f" Completion: {100 - (len(missing)/max_f)*100:.1f}% | Missing: {len(missing)}")
    if missing: print(f" Missing: {missing}")

if __name__ == "__main__":
    run_infallible_healer()