import cv2
import numpy as np
import os
import json
import shutil
import csv

# --- TEST CONFIGURATION ---
TEST_ZONES = [
    {"ds": "0", "start_idx": 2550, "end_idx": 4000},  # F34-42
    {"ds": "0", "start_idx": 7500, "end_idx": 9500},  # F63-70
    {"ds": "0", "start_idx": 17200, "end_idx": 25200}, # F101-109
    {"ds": "1", "start_idx": 22800, "end_idx": 25270}, 
    {"ds": "2", "start_idx": 24000, "end_idx": 25270}
]

# --- BOSS DATA & HUD CONSTANTS (REQUIRED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
SURGICAL_OUT = "Surgical_Proof"

def run_profile_auditor():
    if not os.path.exists(SURGICAL_OUT): os.makedirs(SURGICAL_OUT)
    
    global_report = []

    for zone_cfg in TEST_ZONES:
        ds_id = zone_cfg["ds"]
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(buffer_path): continue
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        range_label = f"Run_{ds_id}_frames_{zone_cfg['start_idx']}_{zone_cfg['end_idx']}"
        ds_out = os.path.join(SURGICAL_OUT, range_label)
        if os.path.exists(ds_out): shutil.rmtree(ds_out)
        os.makedirs(ds_out)

        print(f"\n--- AUDITING {range_label} ---")
        
        start_idx = zone_cfg['start_idx']
        prev_img = cv2.imread(os.path.join(buffer_path, frames[start_idx]), 0)
        prev_h = prev_img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        prev_c = prev_img[CENTER_ROI[0]:CENTER_ROI[1], CENTER_ROI[2]:CENTER_ROI[3]]
        
        anchor_h = prev_h.copy()
        last_anc_diff = 0
        found_count = 0

        for i in range(start_idx + 1, min(zone_cfg['end_idx'], len(frames) - 6)):
            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            curr_c = img[CENTER_ROI[0]:CENTER_ROI[1], CENTER_ROI[2]:CENTER_ROI[3]]
            
            h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
            c_flux = np.mean(cv2.absdiff(curr_c, prev_c))
            anc_diff = np.mean(cv2.absdiff(curr_h, anchor_h))
            
            trigger = False
            # DUAL-PULSE GATE
            if h_flux > 2.1 and c_flux > 1.3:
                # 1. PROFILE CHECK: Is the change horizontally uniform (Banner)?
                diff_map = cv2.absdiff(curr_h, prev_h)
                # Row-wise variance: Banners are 'flat' across rows
                row_sums = np.sum(diff_map, axis=1)
                is_structural = np.std(row_sums) > 15.0 # True floors have high row variance
                
                # 2. PERSISTENCE CHECK (5 frames)
                is_permanent = True
                for offset in range(1, 5):
                    future_h = cv2.imread(os.path.join(buffer_path, frames[i+offset]), 0)[52:76, 100:142]
                    if np.mean(cv2.absdiff(future_h, curr_h)) > 6.0:
                        is_permanent = False; break
                
                # 3. ABSOLUTE VETO
                # If AncDiff is too close to the previous one, it's a stutter (Veto if within 0.1 MSE)
                if is_permanent and is_structural and anc_diff > 2.4 and abs(anc_diff - last_anc_diff) > 0.15:
                    trigger = True
                    found_count += 1
                    anchor_idx = i + 5
                    # Update states
                    anchor_h = cv2.imread(os.path.join(buffer_path, frames[anchor_idx]), 0)[52:76, 100:142]
                    last_anc_diff = anc_diff
                    
                    print(f" [+] Found TR_{found_count} @ Idx {i} (AncDiff: {anc_diff:.2f}, StructScore: {np.std(row_sums):.1f})")
                    shutil.copy2(os.path.join(buffer_path, frames[anchor_idx]), 
                                 os.path.join(ds_out, f"TR_{found_count}_Idx{anchor_idx}_{frames[anchor_idx]}"))
            
            global_report.append({'ds': ds_id, 'idx': i, 'anc_diff': anc_diff, 'triggered': trigger})
            prev_h, prev_c = curr_h, curr_c

    # Save debug log
    with open("surgical_profile_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['ds', 'idx', 'anc_diff', 'triggered'])
        writer.writeheader(); writer.writerows(global_report)

if __name__ == "__main__":
    run_profile_auditor()