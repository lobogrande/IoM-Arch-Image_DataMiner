import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 18
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

# Define the slots you see are missing in your visual check
MISSING_SLOTS = [4, 7, 12, 19] # ADJUST THESE BASED ON YOUR VISUAL CHECK

def run_specimen_recovery():
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}

    img = cv2.imread(os.path.join(run_path, f"F{TARGET_FLOOR}_{seq[TARGET_FLOOR]['frame']}"))
    
    out_dir = "Specimen_Audit"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    print(f"--- Recovering {len(MISSING_SLOTS)} Specimen for Analysis ---")

    for slot in MISSING_SLOTS:
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        
        # Grab a slightly larger ROI (64x64) to see the context/noise
        crop = img[cy-32:cy+32, cx-32:cx+32]
        
        # Apply a high-contrast filter to one half for visual debugging
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Stack raw vs enhanced
        comparison = np.hstack((crop, enhanced))
        
        cv2.imwrite(os.path.join(out_dir, f"F18_Slot{slot}_Analysis.png"), comparison)
        print(f" [+] Exported Slot {slot}")

if __name__ == "__main__":
    run_specimen_recovery()