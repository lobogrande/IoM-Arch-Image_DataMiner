import cv2
import numpy as np
import os
import json
import shutil
import csv

# --- TEST CONFIGURATION ---
TEST_ZONES = [
    {"ds": "0", "start_idx": 2550, "end_idx": 4000},  
    {"ds": "0", "start_idx": 7450, "end_idx": 9500},  
    {"ds": "0", "start_idx": 17200, "end_idx": 25200}, 
    {"ds": "1", "start_idx": 22800, "end_idx": 25270}, 
    {"ds": "2", "start_idx": 24000, "end_idx": 25270}
]

# --- MASTER CONSTANTS ---
HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
SURGICAL_OUT = "Surgical_Proof_Final"

def run_final_discovery():
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

        print(f"\n--- FINAL DISCOVERY AUDIT: {label} ---")
        
        start_idx = zone['start_idx']
        prev_img = cv2.imread(os.path.join(buffer_path, frames[start_idx]), 0)
        prev_h = prev_img[52:76, 100:142]
        prev_c = prev_img[230:246, 250:281]
        
        anchor_h = prev_h.copy()
        found_count = 0
        last_found_idx = -99

        for i in range(start_idx + 1, min(zone['end_idx'], len(frames) - 6)):
            # 1. REFRACTORY LOCK: Stop stutters
            if i < last_found_idx + 8:
                continue

            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[52:76, 100:142]
            curr_c = img[230:246, 250:281]
            
            h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
            c_flux = np.mean(cv2.absdiff(curr_c, prev_c))
            
            # 2. ADAPTIVE PULSE: Catch subtle shifts
            if h_flux > 1.6 and c_flux > 1.1:
                # 3. STRUCTURAL FILTER: Reject uniform banners
                row_sums = np.sum(cv2.absdiff(curr_h, prev_h), axis=1)
                struct_score = np.std(row_sums)
                
                # 4. IDENTITY VETO: Is this frame visually identical to the last floor?
                # Re-reading anchor specifically to avoid drift
                res = cv2.matchTemplate(curr_h, anchor_h, cv2.TM_CCORR_NORMED)
                identity_score = res.max()
                
                if struct_score > 85.0 and identity_score < 0.994:
                    found_count += 1
                    last_found_idx = i
                    anchor_idx = i + 4
                    
                    # Lock New Anchor
                    anchor_h = cv2.imread(os.path.join(buffer_path, frames[anchor_idx]), 0)[52:76, 100:142]
                    print(f" [+] TR_{found_count} @ Idx {i} (Identity: {identity_score:.4f}, Score: {struct_score:.1f})")
                    shutil.copy2(os.path.join(buffer_path, frames[anchor_idx]), 
                                 os.path.join(ds_out, f"TR_{found_count}_Idx{anchor_idx}_{frames[anchor_idx]}"))
            
            prev_h, prev_c = curr_h, curr_c

if __name__ == "__main__":
    run_final_discovery()