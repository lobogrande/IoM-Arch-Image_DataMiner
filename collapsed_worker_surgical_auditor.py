import cv2
import numpy as np
import os
import json
import shutil
import csv

# --- TEST CONFIGURATION (RESTRICTED FOR MULTIPLE ZONES) ---
# Now using a list to allow multiple segments per dataset
TEST_ZONES = [
    {"ds": "0", "start_idx": 2550, "end_idx": 4000},  # F34-42 (Skipped 39 issue)
    {"ds": "0", "start_idx": 7500, "end_idx": 9500},  # F63-70 (Skipped floors issue)
    {"ds": "0", "start_idx": 17200, "end_idx": 25200}, # F101-103 (Double 103 issue)
    {"ds": "1", "start_idx": 22800, "end_idx": 25270}, 
    {"ds": "2", "start_idx": 24000, "end_idx": 25270}
]

HEADER_ROI = (52, 76, 100, 142)
CENTER_ROI = (230, 246, 250, 281)
SURGICAL_OUT = "Surgical_Proof"

def run_multizone_audit():
    if not os.path.exists(SURGICAL_OUT): os.makedirs(SURGICAL_OUT)
    if os.path.exists("surgical_debug_log.csv"): os.remove("surgical_debug_log.csv")
    
    for zone_cfg in TEST_ZONES:
        ds_id = zone_cfg["ds"]
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.listdir(buffer_path): continue
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        
        # Create a unique subfolder for this specific range
        range_label = f"Run_{ds_id}_frames_{zone_cfg['start_idx']}_{zone_cfg['end_idx']}"
        ds_out = os.path.join(SURGICAL_OUT, range_label)
        if os.path.exists(ds_out): shutil.rmtree(ds_out)
        os.makedirs(ds_out)

        print(f"\n--- AUDITING {range_label} ---")
        
        start_idx = zone_cfg['start_idx']
        prev_img = cv2.imread(os.path.join(buffer_path, frames[start_idx]), 0)
        prev_h = prev_img[52:76, 100:142]
        prev_c = prev_img[230:246, 250:281]
        
        anchor_h = prev_h.copy()
        found_in_zone = 0
        report = []

        for i in range(start_idx + 1, min(zone_cfg['end_idx'], len(frames) - 5)):
            img = cv2.imread(os.path.join(buffer_path, frames[i]), 0)
            curr_h = img[52:76, 100:142]
            curr_c = img[230:246, 250:281]
            
            h_flux = np.mean(cv2.absdiff(curr_h, prev_h))
            c_flux = np.mean(cv2.absdiff(curr_c, prev_c))
            diff_from_anchor = np.mean(cv2.absdiff(curr_h, anchor_h))
            
            trigger = False
            # REFINED TRIGGER: Lower h_flux for missed transitions
            if h_flux > 2.1 and c_flux > 1.3:
                # REFINED PERSISTENCE: 4 frames to kill Crosshair Fairy noise
                is_permanent = True
                for offset in range(1, 5):
                    future_h = cv2.imread(os.path.join(buffer_path, frames[i+offset]), 0)[52:76, 100:142]
                    # If any future frame moves too fast (>7.0), it's a banner flicker
                    if np.mean(cv2.absdiff(future_h, curr_h)) > 7.0:
                        is_permanent = False; break
                
                # ADAPTIVE GATE: 2.5 MSE to catch subtle shifts like F39
                if is_permanent and diff_from_anchor > 2.5:
                    trigger = True
                    found_in_zone += 1
                    anchor_idx = i + 5
                    # Lock new anchor
                    anchor_h = cv2.imread(os.path.join(buffer_path, frames[anchor_idx]), 0)[52:76, 100:142]
                    
                    print(f" [+] Found TR_{found_in_zone} @ Idx {i} (Flux: {h_flux:.2f}, AncDiff: {diff_from_anchor:.2f})")
                    shutil.copy2(os.path.join(buffer_path, frames[anchor_idx]), 
                                 os.path.join(ds_out, f"TR_{found_in_zone}_Idx{anchor_idx}_{frames[anchor_idx]}"))
            
            report.append({'ds': ds_id, 'idx': i, 'h_flux': h_flux, 'anc_diff': diff_from_anchor, 'triggered': trigger})
            prev_h, prev_c = curr_h, curr_c

        # Append this zone's data to the global debug log
        with open("surgical_debug_log.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=['ds', 'idx', 'h_flux', 'anc_diff', 'triggered'])
            if f.tell() == 0: writer.writeheader()
            writer.writerows(report)

if __name__ == "__main__":
    run_multizone_audit()