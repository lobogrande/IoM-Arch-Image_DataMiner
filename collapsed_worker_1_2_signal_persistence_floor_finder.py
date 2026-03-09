import cv2
import numpy as np
import os
import json
import shutil

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. MASTER CONSTANTS (VERIFIED HUD/ROI) ---
HEADER_ROI = (52, 76, 100, 142)  # Stage Number
GRID_ROI = (250, 550, 50, 450)    # Main Mining Grid
# NEW PROTECTED OUTPUT FOLDER
COLLAPSED_OUT = "Collapsed_Survey_Results"

def run_archaeological_surveyor():
    if not os.path.exists(COLLAPSED_OUT): os.makedirs(COLLAPSED_OUT)
    
    # Target only dataset 0 for final calibration
    datasets = ["0"] 
    for ds_id in datasets:
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(buffer_path): continue
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        ds_out = os.path.join(COLLAPSED_OUT, f"Run_{ds_id}")
        if os.path.exists(ds_out): shutil.rmtree(ds_out)
        os.makedirs(ds_out)

        print(f"\n--- ARCHAEOLOGICAL SURVEY RUN {ds_id} ---")
        
        # Init state
        sequence = [{'idx': 0, 'floor': 1, 'frame': frames[0]}]
        first_img = cv2.imread(os.path.join(buffer_path, frames[0]), 0)
        anchor_h = first_img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        
        prev_h = anchor_h.copy()
        prev_g = first_img[GRID_ROI[0]:GRID_ROI[1], GRID_ROI[2]:GRID_ROI[3]]
        
        current_floor = 1
        last_found_idx = 0
        
        i = 1
        while i < len(frames) - 10:
            if i < last_found_idx + 8: # Temporal refractory period
                i += 1; continue

            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            curr_g = img[GRID_ROI[0]:GRID_ROI[1], GRID_ROI[2]:GRID_ROI[3]]
            
            # 1. DUAL-SYNC PULSE: Stage moves AND Grid resets
            h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
            g_flux = np.mean(cv2.absdiff(curr_g, prev_g))
            
            # 5.5 Grid Flux is the definitive 'Board Reset' marker
            if h_flux > 1.5 and g_flux > 5.5:
                # 2. IDENTITY VETO: Confirm settled state is actually new
                # Look ahead 4 frames to find the cleanest stable state
                best_std = 999; best_idx = i
                for offset in range(2, 6):
                    check_idx = i + offset
                    check_h = cv2.imread(os.path.join(buffer_path, frames[check_idx]), 0)[52:76, 100:142]
                    c_std = np.std(check_h)
                    if c_std < best_std:
                        best_std = c_std; best_idx = check_idx
                
                settled_h = cv2.imread(os.path.join(buffer_path, frames[best_idx]), 0)[52:76, 100:142]
                res = cv2.matchTemplate(settled_h, anchor_h, cv2.TM_CCORR_NORMED)
                identity = res.max()
                
                # Steel Gate: Structural Identity Veto (0.985)
                if identity < 0.985:
                    current_floor += 1
                    last_found_idx = best_idx
                    sequence.append({'idx': last_found_idx, 'floor': current_floor, 'frame': frames[last_found_idx]})
                    
                    if current_floor % 10 == 0: 
                        print(f"  Confirmed F{current_floor} @ Idx {last_found_idx} (Grid Flux: {g_flux:.2f})")
                    
                    anchor_h = settled_h.copy()
                    i = last_found_idx + 1
                    prev_h = anchor_h.copy()
                    prev_g = cv2.imread(os.path.join(buffer_path, frames[last_found_idx]), 0)[GRID_ROI[0]:GRID_ROI[1], GRID_ROI[2]:GRID_ROI[3]]
                    continue
            
            prev_h, prev_g = curr_h, curr_g
            i += 1

        # Final Archival
        with open(f"surveyed_sequence_run_{ds_id}.json", 'w') as f:
            json.dump(sequence, f, indent=4)
        for entry in sequence:
            shutil.copy2(os.path.join(buffer_path, entry['frame']), 
                         os.path.join(ds_out, f"F{entry['floor']}_{entry['frame']}"))

if __name__ == "__main__":
    run_archaeological_surveyor()