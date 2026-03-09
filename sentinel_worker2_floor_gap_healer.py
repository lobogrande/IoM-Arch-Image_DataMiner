import cv2
import numpy as np
import os
import json
import sys
import shutil

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. CONSTANTS ---
DATASETS = ["0", "1", "2", "3", "4"]
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
FINAL_CONSENSUS_DIR = "Final_Consensus_Images"

def run_final_sentinel():
    if not os.path.exists(FINAL_CONSENSUS_DIR): os.makedirs(FINAL_CONSENSUS_DIR)

    for ds_id in DATASETS:
        json_file = f"milestones_run_{ds_id}.json"
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(json_file): continue
        
        with open(json_file, 'r') as f: anchors = json.load(f)
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        
        ds_final_path = os.path.join(FINAL_CONSENSUS_DIR, f"Run_{ds_id}")
        if os.path.exists(ds_final_path): shutil.rmtree(ds_final_path)
        os.makedirs(ds_final_path)

        print(f"\n--- PROCESSING DATASET {ds_id} ---")
        
        final_list = []
        for i in range(len(anchors) - 1):
            final_list.append(anchors[i])
            count = anchors[i+1]['floor'] - anchors[i]['floor'] - 1
            if count > 0:
                print(f" Healing Gap: F{anchors[i]['floor']} -> F{anchors[i+1]['floor']} ({count} floors)")
                healed = solve_gap_absolute_verbose(buffer_path, frames, anchors[i], anchors[i+1], count)
                final_list.extend(healed)

        final_list.append(anchors[-1])
        final_list.sort(key=lambda x: x['idx'])
        
        # Save merged JSON for Worker 3
        with open(f"final_sequence_run_{ds_id}.json", 'w') as f:
            json.dump(final_list, f, indent=4)

        print(f" Dataset Complete. Sequence length: {len(final_list)}")
        print(f" Migrating unified images to {ds_final_path}...")
        for entry in final_list:
            src = os.path.join(buffer_path, entry['frame'])
            dst = os.path.join(ds_final_path, f"F{entry['floor']}_{entry['frame']}")
            shutil.copy2(src, dst)
            # HUD logic commented out
            # img = cv2.imread(dst)
            # cv2.rectangle(img, (100, 52), (142, 76), (0, 255, 0), 2)
            # cv2.imwrite(dst, img)

def solve_gap_absolute_verbose(path, frames, anc_s, anc_e, count):
    healed = []
    prev_h = cv2.imread(os.path.join(path, frames[anc_s['idx']]), 0)[52:76, 100:142]
    prev_c = cv2.imread(os.path.join(path, frames[anc_s['idx']]), 0)[230:246, 250:281]
    
    found = 0
    for i in range(anc_s['idx'] + 1, anc_e['idx'] - 5):
        if found >= count: break
        img = cv2.imread(os.path.join(path, frames[i]), 0)
        curr_h = img[52:76, 100:142]
        curr_c = img[230:246, 250:281]
        
        # Signal Spike Detection
        if np.mean(cv2.absdiff(curr_h, prev_h)) > 2.2 and np.mean(cv2.absdiff(curr_c, prev_c)) > 1.5:
            # 3-Frame Persistence Gate
            is_permanent = True
            for offset in range(1, 4):
                future_h = cv2.imread(os.path.join(path, frames[i+offset]), 0)[52:76, 100:142]
                if np.mean(cv2.absdiff(future_h, curr_h)) > 5.0:
                    is_permanent = False; break
            
            if is_permanent:
                found += 1
                target_f = anc_s['floor'] + found
                anchor_idx = i + 5
                healed.append({'idx': anchor_idx, 'floor': target_f, 'frame': frames[anchor_idx]})
                
                # RESTORED LOGGING
                print(f"   [F{target_f}] Pulse detected @ {i} -> Anchored @ {anchor_idx}")
                
                prev_h = cv2.imread(os.path.join(path, frames[anchor_idx]), 0)[52:76, 100:142]
                prev_c = cv2.imread(os.path.join(path, frames[anchor_idx]), 0)[230:246, 250:281]
                continue
        prev_h, prev_c = curr_h, curr_c
    return healed

if __name__ == "__main__":
    run_final_sentinel()