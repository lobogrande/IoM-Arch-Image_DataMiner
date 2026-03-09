import cv2
import numpy as np
import os
import json
import sys
import shutil
import re
import time

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. MASTER CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
DIGITS_DIR = "digits"
BASE_HEAL_DIR = "Pass2_Evidence"
HEADER_ROI = (52, 76, 100, 142)
GRID_ROI = (250, 500, 50, 450) # Region for visual state hashing

def run_pincer_sentinel():
    digit_map = load_digit_map_pincer()
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
        print(f"\n--- PINCER SENTINEL RUN {ds_id} ---")
        
        final_consensus = []
        used_state_hashes = set()

        for i in range(len(anchors) - 1):
            final_consensus.append(anchors[i])
            # Register anchor state hash
            used_state_hashes.add(get_board_hash(buffer_path, anchors[i]['frame']))
            
            s_f, e_f = anchors[i]['floor'], anchors[i+1]['floor']
            if (e_f - s_f) > 1:
                print(f" Healing Gap: F{s_f}->F{e_f}...")
                healed = solve_gap_with_pincer(buffer_path, frames, anchors[i], anchors[i+1], digit_map, used_state_hashes)
                final_consensus.extend(healed)
                for h in healed: 
                    save_healed_evidence(buffer_path, h, heal_path)
                    used_state_hashes.add(get_board_hash(buffer_path, h['frame']))

        final_consensus.append(anchors[-1])
        final_consensus.sort(key=lambda x: x['idx'])
        with open(f"healed_consensus_run_{ds_id}.json", 'w') as f:
            json.dump(final_consensus, f, indent=4)
        perform_final_audit(ds_id, final_consensus)

def solve_gap_with_pincer(path, frames, anc_s, anc_e, digit_map, used_hashes):
    """Iteratively identifies floors by looking for stability plateaus between resets"""
    healed = []
    # Identify visual end of start-anchor
    wilderness_start = find_visual_break_pincer(path, frames, anc_s, anc_e['idx'], direction=1)
    
    num_missing = anc_e['floor'] - anc_s['floor'] - 1
    current_search_start = wilderness_start
    
    for target in range(anc_s['floor'] + 1, anc_e['floor']):
        found_m = None
        
        # 1. OCR SEARCH (Priority)
        for i in range(current_search_start, anc_e['idx']):
            gray = cv2.imread(os.path.join(path, frames[i]), 0)
            roi = gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            
            # IDENTITY VETO (7 vs 8)
            val = get_bitwise_structural_pincer(roi, digit_map, 175, 0.75)
            if val == target:
                # Require 3 frames of stability
                stability_check = True
                for s_offset in range(1, 3):
                    next_gray = cv2.imread(os.path.join(path, frames[i + s_offset]), 0)
                    if get_bitwise_structural_pincer(next_gray[52:76, 100:142], digit_map, 175, 0.72) != target:
                        stability_check = False; break
                
                if stability_check:
                    # Final check: is this a repeat frame?
                    if get_board_hash(path, frames[i]) not in used_hashes:
                        found_m = {'idx': i, 'floor': target, 'frame': frames[i]}
                        break
        
        if not found_m:
            # 2. GRADIENT NOMINATION (Fallback)
            # Instead of midpoint, scan for the most stable plateau *after* the previous floor ends
            print(f"   [F{target}] OCR Fail - Pincering stability gradient...")
            found_m = nominate_via_gradient(path, frames, current_search_start, anc_e['idx'], target, anc_e['floor']-target, used_hashes)

        print(f"   [F{target}] Locked: {found_m['frame']} (Idx: {found_m['idx']})")
        sys.stdout.flush()
        healed.append(found_m)
        
        # 3. PINCER MARCH: Find where this healed floor visual signature ends
        current_search_start = find_visual_break_pincer(path, frames, found_m, anc_e['idx'], direction=1)
        
    return healed

def find_visual_break_pincer(path, frames, milestone, limit, direction):
    img = cv2.imread(os.path.join(path, milestone['frame']), 0)
    sig = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    rng = range(milestone['idx'] + direction, limit, direction)
    break_idx = milestone['idx']
    for i in rng:
        curr = cv2.imread(os.path.join(path, frames[i]), 0)[52:76, 100:142]
        if np.mean(cv2.absdiff(curr, sig)) > 15: # Stability break
            break_idx = i; break
    return break_idx + 1

def nominate_via_gradient(path, frames, start, end, floor, remaining, used_hashes):
    """Hunts for a stable board state that differs from previous hashes"""
    gap_size = end - start
    step_size = gap_size // (remaining + 1)
    
    best_frame = start + (step_size // 2)
    # Search for local stability minimum within this segment
    min_variance = 999999
    
    for i in range(start + 2, min(start + step_size, end - 2)):
        # Check board reset spike followed by plateau
        gray = cv2.imread(os.path.join(path, frames[i]), 0)
        grid = gray[GRID_ROI[0]:GRID_ROI[1], GRID_ROI[2]:GRID_ROI[3]]
        
        # Simple variance as stability metric
        var = np.var(grid)
        curr_hash = get_board_hash(path, frames[i])
        
        if var < min_variance and curr_hash not in used_hashes:
            min_variance = var
            best_frame = i
            
    return {'idx': best_frame, 'floor': floor, 'frame': frames[best_frame]}

def get_board_hash(path, frame_name):
    # Generates a simple hash for duplicate detection
    img = cv2.imread(os.path.join(path, frame_name), 0)
    grid = cv2.resize(img[250:500, 50:450], (16, 16))
    return hash(grid.tobytes())

def get_bitwise_structural_pincer(roi, digit_map, thresh, min_conf):
    h, w = roi.shape[:2]
    _, bin_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t_bin in temps:
            res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]):
                    # VETO: Disambiguate 8 from 7/3/9
                    if val == 8:
                        y_mid, y_bot = pt[1] + 10, pt[1] + 18
                        if y_bot < h and bin_roi[y_bot, pt[0]] == 0:
                            matches.append({'x': pt[0], 'val': 9 if bin_roi[y_mid, pt[0]] > 0 else 3})
                        else: matches.append({'x': pt[0], 'val': 8})
                    elif val == 7:
                        y_test = pt[1] + 15
                        if y_test < h and bin_roi[y_test, pt[0]+5] > 0: continue 
                        matches.append({'x': pt[0], 'val': 7})
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
    cv2.putText(img, f"PINCER F{milestone['floor']}", (100, 48), 0, 0.4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(heal_path, f"F{milestone['floor']}_{milestone['frame']}"), img)

def load_digit_map_pincer():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f); v = int(m.group()) if m else None
            if v is not None:
                img = cv2.imread(os.path.join(DIGITS_DIR, f), 0); _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[v].append(b)
    return d_map

def perform_final_audit(run_id, milestones):
    found = sorted(list(set(m['floor'] for m in milestones)))
    max_f = milestones[-1]['floor']; missing = sorted(list(set(range(1, max_f + 1)) - set(found)))
    print(f"--- FINAL AUDIT RUN {run_id} ---")
    print(f" Found: {len(milestones)}/{max_f} | Missing: {missing}")

if __name__ == "__main__":
    run_pincer_sentinel()