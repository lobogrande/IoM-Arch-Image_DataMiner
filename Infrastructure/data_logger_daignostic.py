import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import mss
import time
import csv
import os
from datetime import datetime
from collections import Counter

# --- ROI & CONFIGURATION (v5.7 Stable Baseline) ---
HEADER_ROI = {'top': 281, 'left': 110, 'width': 35, 'height': 16} 
GAME_ROI = {'top': 225, 'left': 10, 'width': 446, 'height': 677}
CSV_FILE = "high_speed_minigame_data.csv"
DIV_FOLDER = "divine_kills"
AUDIT_FOLDER = "floor_audits"

# Thresholds - Precision Tuned for Haste Speed and Blur
ORE_THRESH = 0.65  # Lowered to identify moving ores
SHA_THRESH = 0.75  # Lowered to catch shadows during splash flashes
DIGIT_MATCH_MIN = 0.72 
X_OVERLAP_LIMIT = 4     

for folder in [DIV_FOLDER, AUDIT_FOLDER]:
    if not os.path.exists(folder): os.makedirs(folder)

# --- ORE SPAWN GATES (User Verified Table) ---
SPAWN_GATES = cfg.ORE_RESTRICTIONS

SLOT_COORDS = [(x, y-5) for x, y in [
    (82, 492), (141, 492), (200, 492), (259, 492), (318, 492), (377, 492),
    (82, 551), (141, 551), (200, 551), (259, 551), (318, 551), (377, 551),
    (82, 610), (141, 610), (200, 610), (259, 610), (318, 610), (377, 610),
    (82, 664), (141, 664), (200, 664), (259, 664), (318, 664), (377, 664)
]]

# --- LOADERS ---

def load_digit_templates():
    digits = []
    if not os.path.exists(cfg.DIGIT_DIR): return digits
    for f in sorted(os.listdir(cfg.DIGIT_DIR)):
        if f.endswith(".png"):
            img = cv2.imread(f"digits/{f}", 0)
            if img is not None:
                _, bin_t = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                digits.append((int(f[0]), bin_t))
    return digits

def load_ore_templates():
    templates = {'act': {}, 'sha': {}}
    if not os.path.exists(cfg.TEMPLATE_DIR): return templates
    for f in os.listdir(cfg.TEMPLATE_DIR):
        img = cv2.imread(f"templates/{f}", 0)
        if img is None: continue
        prefix = f.split('_')[0]
        if 'sha' in f:
            if prefix not in templates['sha']: templates['sha'][prefix] = []
            templates['sha'][prefix].append(img)
        elif 'act' in f:
            if prefix not in templates['act']: templates['act'][prefix] = []
            templates['act'][prefix].append(img)
    return templates

digit_temps = load_digit_templates()
ore_templates = load_ore_templates()

class SlotTracker:
    def __init__(self, i):
        self.state = "EMPTY"
        self.tier = None

slots = [SlotTracker(i) for i in range(24)]
current_floor = 0
monitoring_active = False
session_kills = []
session_shadows = 0
session_floors = 0
floor_kills = []
target_census_count = 0

# --- CORE LOGIC ---

def get_floor_spatial(header_gray):
    """Centering logic ensures 61 is not read as 1 by Unit-positioning."""
    norm = cv2.normalize(header_gray, None, 0, 255, cv2.NORM_MINMAX)
    _, h_bin = cv2.threshold(norm, 145, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temp in digit_temps:
        res = cv2.matchTemplate(h_bin, temp, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= DIGIT_MATCH_MIN)
        for pt in zip(*locs[::-1]):
            matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = []
    if matches:
        unique.append(matches[0])
        for i in range(1, len(matches)):
            if matches[i]['x'] - unique[-1]['x'] >= X_OVERLAP_LIMIT: unique.append(matches[i])
    if not unique: return None
    
    CENTER_X = 17
    num = len(unique)
    if num == 1: return unique[0]['val'] if 13 <= unique[0]['x'] <= 21 else None
    if num == 2: return unique[0]['val'] * 10 + unique[1]['val'] if unique[0]['x'] < CENTER_X < unique[1]['x'] else None
    if num == 3: return unique[0]['val'] * 100 + unique[1]['val'] * 10 + unique[2]['val']
    return None

def perform_census(game_gray, floor):
    """Establishes target kill count at start of floor to allow board-state transitions."""
    global session_shadows
    count = 0
    for i, (cx, cy) in enumerate(SLOT_COORDS):
        lx, ly = cx - GAME_ROI['left'], cy - GAME_ROI['top']
        cell = game_gray[max(0, ly-30):ly+30, max(0, lx-30):lx+30]
        
        found_sha = next((t for t, imgs in ore_templates['sha'].items() if np.max(cv2.matchTemplate(cell, imgs[0], cv2.TM_CCOEFF_NORMED)) > SHA_THRESH), None)
        found_act = next((t for t, imgs in ore_templates['act'].items() if t in SPAWN_GATES and (SPAWN_GATES[t][0] <= floor <= SPAWN_GATES[t][1]) and np.max(cv2.matchTemplate(cell, imgs[0], cv2.TM_CCOEFF_NORMED)) > ORE_THRESH), None)
        
        if found_sha or found_act:
            count += 1
            if found_sha:
                slots[i].state = "SHADOW"; slots[i].tier = found_sha
                floor_kills.append(found_sha); session_kills.append(found_sha); session_shadows += 1
            elif found_act:
                slots[i].state = "ACTIVE"; slots[i].tier = found_act
    return count

# --- MAIN LOOP ---
print("\n" + "="*35 + "\n PRO LOGGER v11.4: HASTE OPTIMIZED \n" + "="*35)
try:
    target_start = int(input("Enter NEXT floor to log: "))
except:
    exit()

last_detected = None
clear_start_time = 0

with mss.mss() as sct:
    try:
        while True:
            loop_start = time.perf_counter()
            game_raw = np.array(sct.grab(GAME_ROI))
            game_gray = cv2.cvtColor(game_raw, cv2.COLOR_BGRA2GRAY)
            h_gray = cv2.cvtColor(np.array(sct.grab(HEADER_ROI)), cv2.COLOR_BGRA2GRAY)
            detected = get_floor_spatial(h_gray)

            # 1. BUFFERED HANDSHAKE
            if not monitoring_active:
                if (last_detected is not None and detected != last_detected) or (detected == target_start):
                    time.sleep(0.3)
                    monitoring_active = True; current_floor = target_start
                    session_shadows = 0; session_kills = []; floor_kills = []
                    for sl in slots: sl.state = "EMPTY"; sl.tier = None
                    target_census_count = perform_census(game_gray, current_floor)
                    print(f"\n[!] ANCHOR SECURED. Stage {current_floor} (Census: {target_census_count})")
                else:
                    last_detected = detected; print(f"\rWaiting for transition... Currently view: {detected}   ", end=""); continue

            active_count = 0; shadow_count = 0

            # 2. SCAN GRID (State Persistence)
            for i, (cx, cy) in enumerate(SLOT_COORDS):
                s = slots[i]
                if s.state == "SHADOW":
                    shadow_count += 1; continue
                
                lx, ly = cx - GAME_ROI['left'], cy - GAME_ROI['top']
                cell = game_gray[max(0, ly-30):ly+30, max(0, lx-30):lx+30]
                
                found_sha = any(np.max(cv2.matchTemplate(cell, imgs[0], cv2.TM_CCOEFF_NORMED)) > SHA_THRESH for imgs in ore_templates['sha'].values())

                if found_sha:
                    if s.tier:
                        session_kills.append(s.tier); floor_kills.append(s.tier)
                        with open(CSV_FILE, 'a', newline='') as f:
                            csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), current_floor, s.tier])
                        if 'div' in s.tier:
                            cv2.imwrite(f"{DIV_FOLDER}/{s.tier}_F{current_floor}_S{i}.png", cell)
                    s.state = "SHADOW"; shadow_count += 1; session_shadows += 1
                else:
                    if s.state != "ACTIVE":
                        found_tier = next((t for t, imgs in ore_templates['act'].items() if t in SPAWN_GATES and (SPAWN_GATES[t][0] <= current_floor <= SPAWN_GATES[t][1]) and np.max(cv2.matchTemplate(cell, imgs[0], cv2.TM_CCOEFF_NORMED)) > ORE_THRESH), None)
                        if found_tier: s.state = "ACTIVE"; s.tier = found_tier
                    if s.state == "ACTIVE": active_count += 1

            # 3. TRANSITION GATE (Multi-Signal)
            ocr_change = (detected is not None and detected != current_floor)
            board_done = (target_census_count > 0 and shadow_count >= target_census_count)
            
            if board_done:
                if clear_start_time == 0: clear_start_time = time.time()
                if (time.time() - clear_start_time) < 0.3: board_done = False
            else: clear_start_time = 0

            if ocr_change or board_done:
                time.sleep(0.4) # Temporal Anchor
                current_game = np.array(sct.grab(GAME_ROI))
                target = get_floor_spatial(cv2.cvtColor(np.array(sct.grab(HEADER_ROI)), cv2.COLOR_BGRA2GRAY))
                
                # Logical drift protection
                if target is None or not (target == current_floor + 1 or target == 1): target = current_floor + 1
                
                print(f"\n[SUMMARY] Stage {current_floor}: {dict(Counter(floor_kills))} | Shadows: {shadow_count}/{target_census_count}")
                if current_floor % 5 == 0:
                    cv2.imwrite(f"{AUDIT_FOLDER}/AUDIT_F{current_floor}.png", cv2.cvtColor(current_game, cv2.COLOR_BGR2RGB))
                
                current_floor = target; session_floors += 1; floor_kills = []; clear_start_time = 0
                for sl in slots: sl.state = "EMPTY"; sl.tier = None
                target_census_count = perform_census(cv2.cvtColor(current_game, cv2.COLOR_BGR2GRAY), current_floor)

            print(f"\rFloor: {current_floor} | Living: {active_count} | Shadows: {shadow_count}/{target_census_count} | Hz: {1.0/(time.perf_counter()-loop_start):.1f}", end="")

    except KeyboardInterrupt:
        pass

# --- SESSION REPORT ---
print("\n\n" + "="*35 + "\n FINAL SESSION REPORT \n" + "="*35)
print(f" Total Floors Processed:  {session_floors}\n Total Shadows Counted:   {session_shadows}\n Total Ores Identified:   {len(session_kills)}")
if session_shadows > 0:
    print(f" Session ID Accuracy:     {(len(session_kills)/session_shadows)*100:.1f}%")
print("\n--- TIER DISTRIBUTION ---")
for tier, count in sorted(Counter(session_kills).items()):
    print(f" {tier:10}: {count}")
print("="*35)