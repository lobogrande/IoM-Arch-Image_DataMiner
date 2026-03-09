import cv2
import numpy as np
import os
import json
import sys
import shutil

# --- MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
BASE_HEAL_DIR = "Pass2_Evidence"

def run_absolute_sentinel():
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
        print(f"\n--- ABSOLUTE SENTINEL RUN {ds_id} ---")
        
        final_consensus = []
        for i in range(len(anchors) - 1):
            final_consensus.append(anchors[i])
            count = anchors[i+1]['floor'] - anchors[i]['floor'] - 1
            
            if count > 0:
                print(f" Gap F{anchors[i]['floor']}->F{anchors[i+1]['floor']}...")
                healed = solve_gap_absolute(buffer_path, frames, anchors[i], anchors[i+1], count)
                final_consensus.extend(healed)
                for h in healed: save_healed_evidence(buffer_path, h, heal_path)

        final_consensus.append(anchors[-1])
        final_consensus.sort(key=lambda x: x['idx'])
        with open(f"healed_consensus_run_{ds_id}.json", 'w') as f:
            json.dump(final_consensus, f, indent=4)

def solve_gap_absolute(path, frames, anc_s, anc_e, count):
    healed = []
    # Load initial signatures
    prev_h = cv2.imread(os.path.join(path, frames[anc_s['idx']]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    prev_c = cv2.imread(os.path.join(path, frames[anc_s['idx']]), 0)[CENTER_ROI[0]:CENTER_ROI[1], CENTER_ROI[2]:CENTER_ROI[3]]
    
    found = 0
    for i in range(anc_s['idx'] + 1, anc_e['idx'] - 5):
        if found >= count: break
            
        img = cv2.imread(os.path.join(path, frames[i]), 0)
        curr_h = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        curr_c = img[CENTER_ROI[0]:CENTER_ROI[1], CENTER_ROI[2]:CENTER_ROI[3]]
        
        # 1. THE PULSE: Did both Header and Center shift?
        h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
        c_flux = np.mean(cv2.absdiff(curr_c, prev_c))
        
        if h_flux > 2.2 and c_flux > 1.5:
            # 2. PERSISTENCE CHECK: Does it stay changed for 3 frames?
            # (Ensures this isn't just a flicker or banner pass-through)
            is_permanent = True
            for offset in range(1, 4):
                future_h = cv2.imread(os.path.join(path, frames[i + offset]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                if np.mean(cv2.absdiff(future_h, curr_h)) > 5.0: # If it's still moving, it's a banner
                    is_permanent = False; break
            
            if is_permanent:
                found += 1
                target_f = anc_s['floor'] + found
                anchor_idx = i + 5 # Deep in the quiet zone
                healed.append({'idx': anchor_idx, 'floor': target_f, 'frame': frames[anchor_idx]})
                print(f"   [F{target_f}] Permanent shift @ {i} -> Anchored @ {anchor_idx}")
                
                # Update baselines
                prev_h = cv2.imread(os.path.join(path, frames[anchor_idx]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                prev_c = cv2.imread(os.path.join(path, frames[anchor_idx]), 0)[CENTER_ROI[0]:CENTER_ROI[1], CENTER_ROI[2]:CENTER_ROI[3]]
                continue
                
        prev_h, prev_c = curr_h, curr_c
        
    return healed

def save_healed_evidence(path, milestone, heal_path):
    img = cv2.imread(os.path.join(path, milestone['frame']))
    cv2.rectangle(img, (100, 52), (142, 76), (0, 255, 255), 2)
    cv2.putText(img, f"ABSOLUTE F{milestone['floor']}", (100, 48), 0, 0.4, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(heal_path, f"F{milestone['floor']}_{milestone['frame']}"), img)

if __name__ == "__main__":
    run_absolute_sentinel()