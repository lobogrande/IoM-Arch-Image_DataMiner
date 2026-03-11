import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 11) # Scan the first 10 floors
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48

# THE MAGIC NUMBERS (Based on your Energy Profile results)
MIN_ENERGY = 500  # Ores must have at least this much structural detail
STRICT_THRESH = 0.80

def run_signature_diagnostic():
    # 1. Load Templates (Ore only)
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (AI_DIM, AI_DIM))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    
    print(f"--- Dual Signature Diagnostic: Floors 1-10 ---")
    print(f"Using Energy Threshold: {MIN_ENERGY}")

    for f_num in TARGET_FLOORS:
        files = [f for f in os.listdir(run_path) if f.startswith(f"F{f_num}_")]
        if not files: continue
        
        raw_img = cv2.imread(os.path.join(run_path, files[0]))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        found_count = 0

        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            roi = gray[cy-24:cy+24, cx-24:cx+24]
            
            # A. Calculate Energy FIRST (The Gatekeeper)
            energy = cv2.Laplacian(roi, cv2.CV_64F).var()
            
            if energy < MIN_ENERGY:
                continue # Skip template matching entirely for "dead" slots
            
            # B. If it has energy, find the best template match
            best_score = 0
            best_name = ""
            for t in ore_templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCOEFF_NORMED)
                score = res.max()
                if score > best_score:
                    best_score = score
                    best_name = t['name']
            
            # Final Verdict
            if best_score > STRICT_THRESH:
                found_count += 1
                cv2.rectangle(raw_img, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
                cv2.putText(raw_img, f"E:{int(energy)}", (cx-22, cy+20), 0, 0.3, (0, 255, 0), 1)

        out_name = f"Signature_Audit_F{f_num}.jpg"
        cv2.imwrite(out_name, raw_img)
        print(f" Floor {f_num:02d}: Identified {found_count} valid ores.")

if __name__ == "__main__":
    run_signature_diagnostic()