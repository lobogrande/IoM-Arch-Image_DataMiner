import cv2
import numpy as np
import os
import json
import sys
import shutil
import re

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. MASTER CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
DIGITS_DIR = "digits"
BASE_HEAL_DIR = "Pass2_Evidence"
HEADER_ROI = (52, 76, 100, 142)

def run_signal_sentinel():
    digit_map = load_digit_map_final()
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
        print(f"\n--- SIGNAL SENTINEL RUN {ds_id} ---")
        
        final_consensus = []
        for i in range(len(anchors) - 1):
            final_consensus.append(anchors[i])
            if (anchors[i+1]['floor'] - anchors[i]['floor']) > 1:
                print(f" Signal Mapping Gap: F{anchors[i]['floor']} -> F{anchors[i+1]['floor']}")
                sys.stdout.flush()
                healed = solve_gap_via_signal(buffer_path, frames, anchors[i], anchors[i+1], digit_map)
                final_consensus.extend(healed)
                for h in healed: save_healed_evidence(buffer_path, h, heal_path)

        final_consensus.append(anchors[-1])
        final_consensus.sort(key=lambda x: x['idx'])
        with open(f"healed_consensus_run_{ds_id}.json", 'w') as f:
            json.dump(final_consensus, f, indent=4)
        perform_final_audit(ds_id, final_consensus)

def solve_gap_via_signal(path, frames, anc_s, anc_e, digit_map):
    """Uses physical pixel flux to find boundaries instead of unreliable OCR"""
    healed = []
    current_search_start = anc_s['idx']
    
    for target in range(anc_s['floor'] + 1, anc_e['floor']):
        # 1. PROFILE PREVIOUS FLOOR END: Identify the 'Cliff'
        # We look for the exact frame where the pixels move (Flux > 3.0)
        cliff_idx = find_flux_cliff(path, frames, current_search_start, anc_e['idx'])
        
        # 2. FIND NEXT PLATEAU: Where pixels stop changing after a reset
        # We target the center of the quiet zone immediately after the cliff
        plateau_idx = find_next_stable_plateau(path, frames, cliff_idx, anc_e['idx'])
        
        # 3. ANCHOR: Use the plateau center as the floor evidence
        found_m = {'idx': plateau_idx, 'floor': target, 'frame': frames[plateau_idx]}
        print(f"   [F{target}] Signal Locked: {found_m['frame']} (Index: {found_m['idx']})")
        sys.stdout.flush()
        
        healed.append(found_m)
        current_search_start = plateau_idx
        
    return healed

def find_flux_cliff(path, frames, start, limit):
    """Identifies the exact frame where the Stage box pixels move"""
    prev_roi = cv2.imread(os.path.join(path, frames[start]), 0)[52:76, 100:142]
    for i in range(start + 1, limit):
        curr_roi = cv2.imread(os.path.join(path, frames[i]), 0)[52:76, 100:142]
        # Match the ~6.8 jump found in the diagnostic CSV
        if np.mean(cv2.absdiff(curr_roi, prev_roi)) > 3.5: 
            return i
        prev_roi = curr_roi
    return start + 1

def find_next_stable_plateau(path, frames, start, limit):
    """Hunts for a window of 3 frames with near-zero movement"""
    for i in range(start, limit - 3):
        jitter = []
        for j in range(2):
            f1 = cv2.imread(os.path.join(path, frames[i+j]), 0)[52:76, 100:142]
            f2 = cv2.imread(os.path.join(path, frames[i+j+1]), 0)[52:76, 100:142]
            jitter.append(np.mean(cv2.absdiff(f1, f2)))
        
        # Mathematically quiet plateau
        if np.mean(jitter) < 0.5: 
            return i + 1 
    return start + 1

def load_digit_map_final():
    d_map = {i: [] for i in range(10)}
    for f in os.listdir(DIGITS_DIR):
        if f.endswith('.png'):
            m = re.search(r'\d', f); v = int(m.group()) if m else None
            if v is not None:
                img = cv2.imread(os.path.join(DIGITS_DIR, f), 0); _, b = cv2.threshold(img, 195, 255, cv2.THRESH_BINARY)
                d_map[v].append(b)
    return d_map

def save_healed_evidence(path, milestone, heal_path):
    img = cv2.imread(os.path.join(path, milestone['frame']))
    cv2.rectangle(img, (100, 52), (142, 76), (255, 0, 0), 2)
    cv2.putText(img, f"SIGNAL F{milestone['floor']}", (100, 48), 0, 0.4, (255, 0, 0), 1)
    cv2.imwrite(os.path.join(heal_path, f"F{milestone['floor']}_{milestone['frame']}"), img)

def perform_final_audit(run_id, milestones):
    found = sorted(list(set(m['floor'] for m in milestones)))
    max_f = milestones[-1]['floor']
    missing = sorted(list(set(range(1, max_f + 1)) - set(found)))
    print(f"--- FINAL AUDIT RUN {run_id} ---")
    print(f" Found: {len(milestones)}/{max_f} | Missing: {missing}")

if __name__ == "__main__":
    run_signal_sentinel()