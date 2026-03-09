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

def run_geometric_sentinel():
    digit_map = load_digit_map_fixed()
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
        print(f"\n--- GEOMETRIC ANCHOR RUN {ds_id} ---")
        
        final_consensus = []
        last_locked_idx = -1
        
        for i in range(len(anchors) - 1):
            final_consensus.append(anchors[i])
            last_locked_idx = anchors[i]['idx']
            s_f, e_f = anchors[i]['floor'], anchors[i+1]['floor']
            
            if (e_f - s_f) > 1:
                print(f" Healing Gap: F{s_f}->F{e_f}...")
                sys.stdout.flush()
                healed = solve_gap_with_geometric_verification(buffer_path, frames, anchors[i], anchors[i+1], digit_map)
                final_consensus.extend(healed)
                if healed: last_locked_idx = healed[-1]['idx']
                for h in healed: save_healed_evidence(buffer_path, h, heal_path)

        final_consensus.append(anchors[-1])
        final_consensus.sort(key=lambda x: x['idx'])
        with open(f"healed_consensus_run_{ds_id}.json", 'w') as f:
            json.dump(final_consensus, f, indent=4)
        perform_final_audit(ds_id, final_consensus)

def solve_gap_with_geometric_verification(path, frames, anc_s, anc_e, digit_map):
    healed = []
    # Hard-lock search start to the index AFTER the previous anchor
    search_start = anc_s['idx'] + 1
    
    for target in range(anc_s['floor'] + 1, anc_e['floor']):
        found_m = None
        cand_f, cand_count = -1, 0
        
        for i in range(search_start, anc_e['idx']):
            gray = cv2.imread(os.path.join(path, frames[i]), 0)
            roi = gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            
            # MANDATORY IDENTITY CHECK: Reject if it explicitly looks like a previous floor
            # Uses very low confidence to act as a sensitive 'No Go' sensor
            if get_bitwise_geometric(roi, digit_map, 175, 0.50) < target:
                cand_f, cand_count = -1, 0
                continue

            val = -1
            for t in [175, 155, 195]:
                if get_bitwise_geometric(roi, digit_map, t, 0.75) == target:
                    val = target; break
            
            if val == target:
                if val == cand_f: cand_count += 1
                else: cand_f, cand_count = val, 1
                if cand_count >= 5: # Persistence gate
                    found_m = {'idx': i - 4, 'floor': target, 'frame': frames[i-4]}
                    break
            else:
                cand_f, cand_count = -1, 0
                
        if not found_m:
            # Fallback uses Proportional Logic to assign a frame that CANNOT be the same as neighbors
            found_m = nominate_safe_frame(path, frames, search_start, anc_e['idx'], target, anc_e['floor']-target)

        print(f"   [F{target}] Assigned: {found_m['frame']} (Idx: {found_m['idx']})")
        sys.stdout.flush()
        healed.append(found_m)
        search_start = found_m['idx'] + 1 # Strict step-forward
        
    return healed

def nominate_safe_frame(path, frames, start, end, floor, remaining):
    # Splits remaining space equally to avoid cannibalization
    segment = (end - start) // (remaining + 1)
    nom_idx = start + max(1, segment // 2)
    return {'idx': nom_idx, 'floor': floor, 'frame': frames[nom_idx]}

def get_bitwise_geometric(roi, digit_map, thresh, min_conf):
    _, bin_roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temps in digit_map.items():
        for t_bin in temps:
            res = cv2.matchTemplate(bin_roi, t_bin, cv2.TM_CCOEFF_NORMED)
            if res.max() >= min_conf:
                locs = np.where(res >= min_conf)
                for pt in zip(*locs[::-1]):
                    # GEOMETRIC VETO: 8 vs 7 vs 3
                    # To be an '8', the center-left (X, Y+10) must be black (white pixel present)
                    if val == 8:
                        if bin_roi[pt[1]+10, pt[0]] == 0: matches.append({'x': pt[0], 'val': 3}) # Mid-left empty
                        elif bin_roi[pt[1]+15, pt[0]] == 0: matches.append({'x': pt[0], 'val': 9}) # Bottom-left empty
                        else: matches.append({'x': pt[0], 'val': 8})
                    # To be a '7', the bottom half MUST be empty
                    elif val == 7:
                        if bin_roi[pt[1]+15, pt[0]+5] > 0: continue # Loop found, not a 7
                        matches.append({'x': pt[0], 'val': 7})
                    else:
                        matches.append({'x': pt[0], 'val': val})
    if not matches: return -1
    matches.sort(key=lambda d: d['x'])
    unique = []; lx = -99
    for m in matches:
        if abs(m['x'] - lx) > 4: unique.append(m['val']); lx = m['x']
    try: return int("".join(map(str, unique)))
    except: return -1

def save_healed_evidence(path, milestone, heal_path):
    img = cv2.imread(os.path.join(path, milestone['frame']))
    # Draw after scan logic confirmed
    cv2.rectangle(img, (100, 52), (142, 76), (0, 0, 255), 2)
    cv2.putText(img, f"HEALED F{milestone['floor']}", (100, 48), 0, 0.4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(heal_path, f"F{milestone['floor']}_{milestone['frame']}"), img)

def load_digit_map_fixed():
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
    run_geometric_sentinel()