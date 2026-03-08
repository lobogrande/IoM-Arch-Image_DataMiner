import cv2
import numpy as np
import os
import csv
import sys
import json
import shutil

# --- 1. THE GROUND TRUTH ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'},
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'},
    34: {'tier': 'mixed', 'special': {8: 'myth1', 9: 'myth1', 14: 'myth1', 15: 'myth1',
                   **{i: 'com2' for i in range(24) if i not in [8, 9, 14, 15]}}},
    49: {'tier': 'mixed', 'special': {**{i: 'dirt3' for i in range(0, 6)}, **{i: 'com3' for i in range(6, 12)},
                   **{i: 'rare3' for i in range(12, 18)}, **{i: 'myth2' for i in range(18, 24)}}},
    74: {'tier': 'mixed', 'special': {20: 'div1', 21: 'div1', **{i: 'dirt3' for i in range(24) if i not in [20, 21]}}},
    98: {'tier': 'myth3'},
    99: {'tier': 'mixed', 'special': {**{i: 'com3' for i in [0, 6, 12, 18]}, **{i: 'rare3' for i in [1, 7, 13, 19]},
                   **{i: 'epic3' for i in [2, 8, 14, 20]}, **{i: 'leg3' for i in [3, 9, 15, 21]},
                   **{i: 'myth3' for i in [4, 10, 16, 22]}, **{i: 'div2' for i in [5, 11, 17, 23]}}}
}

ORE_CONSTRAINTS = {
    'dirt1': [1, 11], 'com1': [1, 17], 'rare1': [3, 25], 'epic1': [6, 29],
    'leg1': [12, 31], 'myth1': [20, 34], 'div1': [50, 74],
    'dirt2': [12, 23], 'com2': [18, 28], 'rare2': [26, 35], 'epic2': [30, 41],
    'leg2': [32, 44], 'myth2': [36, 49], 'div2': [75, 99],
    'dirt3': [24, 999], 'com3': [30, 999], 'rare3': [36, 999],
    'epic3': [42, 999], 'leg3': [45, 999], 'myth3': [50, 999], 'div3': [100, 999]
}

# --- 2. CONFIGURATION & COORDINATES ---
BASE_DIR = "forensic_v37_1"
BUFFER_DIR = "capture_buffer"
TEMPLATE_DIR = "templates"
DIGITS_DIR = "digits"
START_IMAGE = "frame_20260306_231742_176023.png"

# Checkpoints
MILESTONE_JSON = os.path.join(BASE_DIR, "checkpoint_pass1.json")
FLOORMAP_JSON = os.path.join(BASE_DIR, "checkpoint_pass2.json")

# Coordinates
AI_COORDS = {i: (100 + (i % 6) * 60, 500 + (i // 6) * 65) for i in range(24)}
HUD_COORDS = {i: (73 + (i % 6) * 59, 268 + (i // 6) * 61) for i in range(24)}
HEADER_ROI = (56, 100, 16, 35) # Y, X, H, W
DIG_STAGE_ROI = (330, 185, 20, 100) # Your identified location for transition text

# --- 3. CORE ENGINES ---
def get_bitwise_floor(gray_roi, digit_map, min_conf=0.85):
    _, bin_h = cv2.threshold(gray_roi, 195, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t in temps:
            res = cv2.matchTemplate(bin_h, t, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]): matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = []
    if matches:
        unique.append(matches[0]['val'])
        for i in range(1, len(matches)):
            if abs(matches[i]['x'] - matches[i-1]['x']) > 6: unique.append(matches[i]['val'])
    return int("".join(map(str, unique))) if unique else -1

def clean_dir(path):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)

# --- 4. AGENTS ---
def scout_pass(frames, start_idx, digit_map):
    if os.path.exists(MILESTONE_JSON): return json.load(open(MILESTONE_JSON, 'r'))
    print("\n--- PASS 1: SCOUTING MILESTONES ---")
    anchors = [{'idx': start_idx, 'floor': 1, 'name': frames[start_idx], 'type': 'Pass1_Scout'}]
    target = 2
    for i in range(start_idx + 1, len(frames)):
        img = cv2.imread(os.path.join(BUFFER_DIR, frames[i]), 0)
        # Scan both Header and Dig-Stage for maximum coverage
        h_val = get_bitwise_floor(img[56:72, 100:135], digit_map, min_conf=0.92)
        d_val = get_bitwise_floor(img[330:350, 185:285], digit_map, min_conf=0.92)
        
        val = max(h_val, d_val)
        sys.stdout.write(f"\r [SCOUT] Scanning: {frames[i]} | Found: {target-1}")
        if val >= target:
            anchors.append({'idx': i, 'floor': val, 'name': frames[i], 'type': 'Pass1_Scout'})
            print(f"\n [SCOUT] Floor {val} at {frames[i]}")
            target = val + 1
    with open(MILESTONE_JSON, 'w') as f: json.dump(anchors, f)
    return anchors

def inquisitor_pass(frames, anchors, digit_map):
    if os.path.exists(FLOORMAP_JSON): return json.load(open(FLOORMAP_JSON, 'r'))
    print("\n--- PASS 2: INQUISITOR RECOVERY ---")
    full_map = []
    for i in range(len(anchors)-1):
        curr, nxt = anchors[i], anchors[i+1]
        full_map.append(curr)
        if nxt['floor'] > curr['floor'] + 1 or nxt['floor'] == -1:
            seeking = curr['floor'] + 1
            for s_idx in range(curr['idx']+1, nxt['idx']):
                sys.stdout.write(f"\r  [INQ] Checking: {frames[s_idx]} | Seeking: {seeking}")
                img = cv2.imread(os.path.join(BUFFER_DIR, frames[s_idx]), 0)
                # Aggressive search in both ROI locations
                val_h = get_bitwise_floor(img[56:72, 100:135], digit_map, min_conf=0.65)
                val_d = get_bitwise_floor(img[330:350, 185:285], digit_map, min_conf=0.65)
                val = max(val_h, val_d)
                if val == seeking:
                    full_map.append({'idx': s_idx, 'floor': val, 'name': frames[s_idx], 'type': 'Pass2_Inquisitor'})
                    print(f"\n  [FOUND] Recovered Floor {val}")
                    seeking += 1
    with open(FLOORMAP_JSON, 'w') as f: json.dump(full_map, f)
    return full_map

def auditor_pass(full_map, templates):
    print("\n--- PASS 3: AUDITOR CENSUS ---")
    with open(os.path.join(BASE_DIR, "FINAL_AUDIT.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Floor", "Slot", "Tier", "Source"])
        for entry in sorted(full_map, key=lambda x: x['idx']):
            if entry['floor'] == -1: continue
            
            ev_path = os.path.join(BASE_DIR, "Evidence_Library", entry['type'])
            if not os.path.exists(ev_path): os.makedirs(ev_path)
            
            # Census Capture (Shifted further to avoid 7-vs-8 desync)
            c_idx = min(entry['idx'] + 20, len(frames)-1)
            raw = cv2.imread(os.path.join(BUFFER_DIR, frames[c_idx]))
            cv2.imwrite(os.path.join(ev_path, f"F{entry['floor']}_Source.jpg"), raw)
            
            ores, hud = perform_constrained_census(raw, entry['floor'], templates)
            for s, t in ores.items(): writer.writerow([entry['name'], entry['floor'], s, t, entry['type']])
            
            hud_path = os.path.join(BASE_DIR, "Audit_HUDs")
            if not os.path.exists(hud_path): os.makedirs(hud_path)
            cv2.imwrite(os.path.join(hud_path, f"F{entry['floor']}_Audit.jpg"), hud)
    print("\n--- AUDIT COMPLETE ---")

def perform_constrained_census(img, floor, templates):
    res = {}; gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); ann = img.copy()
    is_boss = floor in BOSS_DATA
    for i in range(24):
        ax, ay = AI_COORDS[i]; hx, hy = HUD_COORDS[i]
        if is_boss:
            tier = BOSS_DATA[floor]['special'].get(i, BOSS_DATA[floor]['tier']) if 'special' in BOSS_DATA[floor] else BOSS_DATA[floor]['tier']
        else:
            roi = gray[ay-24:ay+24, ax-24:ax+24]
            roi = cv2.equalizeHist(roi) # Contrast Boost
            best_t, best_s = "obscured", 0.0
            for name, t_img in templates.items():
                lims = ORE_CONSTRAINTS.get(name, [0, 0])
                if lims[0] <= floor <= lims[1]:
                    m = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED).max()
                    if m > best_s: best_s, best_t = m, name
            tier = best_t if best_s >= 0.38 else "obscured"
        res[i] = tier
        cv2.rectangle(ann, (hx-24, hy-24), (hx+24, hy+24), (0,255,255), 1)
        # Write text INSIDE at the bottom
        cv2.putText(ann, tier[:6], (hx-22, hy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
    return res, ann

if __name__ == "__main__":
    if not os.path.exists(BASE_DIR): os.makedirs(BASE_DIR)
    start_at = 1
    for arg in sys.argv:
        if "--start_at=" in arg: start_at = int(arg.split("=")[1])
    if start_at <= 1 and os.path.exists(MILESTONE_JSON): os.remove(MILESTONE_JSON)
    if start_at <= 2 and os.path.exists(FLOORMAP_JSON): os.remove(FLOORMAP_JSON)

    # Standard asset loading
    digit_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            v = int(f[0]); d_img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
            _, b = cv2.threshold(d_img, 195, 255, cv2.THRESH_BINARY); digit_map[v].append(b)
    templates = {f.split('.')[0]: cv2.imread(os.path.join(TEMPLATE_DIR, f), 0) for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')}
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(('.png', '.jpg'))])
    start_idx = frames.index(START_IMAGE)

    anchors = scout_pass(frames, start_idx, digit_map)
    full_map = inquisitor_pass(frames, anchors, digit_map)
    auditor_pass(full_map, templates)