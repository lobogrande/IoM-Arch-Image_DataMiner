import cv2
import numpy as np
import os
import json
import shutil
import csv

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. MASTER CONSTANTS ---
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
SURGICAL_OUT = "Surgical_Proof_Hardened"

# --- 3. TEST ZONES (ALL REGIONS RESTORED) ---
TEST_ZONES = [
    {"ds": "0", "start_idx": 2550, "end_idx": 4000},  # F34-42
    {"ds": "0", "start_idx": 7450, "end_idx": 9500},  # F63-71 (F69 Gap Fix)
    {"ds": "0", "start_idx": 17200, "end_idx": 25200}, # F101-109 (F103 Dupe Fix)
    {"ds": "1", "start_idx": 22800, "end_idx": 25270}, 
    {"ds": "2", "start_idx": 24000, "end_idx": 25270}
]

def run_hardened_discovery():
    if not os.path.exists(SURGICAL_OUT): os.makedirs(SURGICAL_OUT)
    
    for zone in TEST_ZONES:
        ds_id = zone["ds"]
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(buffer_path): continue
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        label = f"Run_{ds_id}_f{zone['start_idx']}_{zone['end_idx']}"
        ds_out = os.path.join(SURGICAL_OUT, label)
        if os.path.exists(ds_out): shutil.rmtree(ds_out)
        os.makedirs(ds_out)

        print(f"\n--- HARDENED AUDIT: {label} ---")
        
        start_idx = zone['start_idx']
        prev_img = cv2.imread(os.path.join(buffer_path, frames[start_idx]), 0)
        prev_h = prev_img[52:76, 100:142]
        prev_c = prev_img[230:246, 250:281]
        
        anchor_h = prev_h.copy()
        found_count = 0
        last_found_idx = -99

        for i in range(start_idx + 1, min(zone['end_idx'], len(frames) - 6)):
            if i < last_found_idx + 6: continue # Refractory buffer

            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[52:76, 100:142]; curr_c = img[230:246, 250:281]
            
            h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
            c_flux = np.mean(cv2.absdiff(curr_c, prev_c))
            
            # TRIGGER GATE
            if h_flux > 1.5 and c_flux > 1.1:
                # 1. Structural Row-Variance
                row_sums = np.sum(cv2.absdiff(curr_h, prev_h), axis=1)
                struct_score = np.std(row_sums)
                
                # 2. Persistence Check
                is_permanent = True
                for offset in range(1, 4):
                    future_h = cv2.imread(os.path.join(buffer_path, frames[i+offset]), 0)[52:76, 100:142]
                    if np.mean(cv2.absdiff(future_h, curr_h)) > 6.0:
                        is_permanent = False; break
                
                # 3. IDENTITY VETO: High-precision template match
                res = cv2.matchTemplate(curr_h, anchor_h, cv2.TM_CCORR_NORMED)
                identity = res.max()
                
                # 4. LITERAL DELTA: MSE against anchor
                anc_mae = np.mean(cv2.absdiff(curr_h, anchor_h))
                
                if is_permanent and struct_score > 100.0 and identity < 0.992 and anc_mae > 2.8:
                    found_count += 1
                    last_found_idx = i
                    anchor_idx = i + 4
                    
                    anchor_h = cv2.imread(os.path.join(buffer_path, frames[anchor_idx]), 0)[52:76, 100:142]
                    print(f" [+] HARDENED TR_{found_count} @ Idx {i} (Identity: {identity:.4f}, Delta: {anc_mae:.2f})")
                    shutil.copy2(os.path.join(buffer_path, frames[anchor_idx]), 
                                 os.path.join(ds_out, f"TR_{found_count}_Idx{anchor_idx}_{frames[anchor_idx]}"))
            
            prev_h, prev_c = curr_h, curr_c

if __name__ == "__main__":
    run_hardened_discovery()