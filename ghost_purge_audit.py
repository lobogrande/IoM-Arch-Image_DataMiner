import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
# Focusing ONLY on the identified problematic "ghost" floors
PROBLEM_FLOORS = [2, 5, 6, 14, 24, 37]
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
TIMESTAMP = datetime.now().strftime('%H%M')
OUTPUT_DIR = f"diagnostic_results/Part1_GhostPurge_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES (Tightened for exclusion)
D_GATE = 6      # Background difference
O_GATE = 0.72   # Raised slightly to reject weak "text" matches
U_GATE = 0.80   # UI Template match sensitivity
DELTA_GATE = 0.08 # MUST beat background by this much to be an ore

def run_ghost_purge_audit():
    # 1. Load Assets
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    ui_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_ui")]
    
    all_ore_t = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            all_ore_t.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}

    print(f"--- Running Part 1: Ghost Purge (Targeting Slots 2 & 3) ---")

    for f_num in PROBLEM_FLOORS:
        if f_num not in seq: continue
        f_name = seq[f_num]['frame']
        raw_img = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{f_name}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray[y1:y1+48, x1:x1+48]
            
            # --- SURGICAL UI MASK (Slots 1-4) ---
            mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6:
                cv2.rectangle(mask, (5, 18), (43, 45), 255, -1) # Ignore the text zone
            else:
                cv2.circle(mask, (24, 24), 16, 255, -1)

            # --- GATE 1: OCCUPANCY ---
            bg_diff = min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_t])
            if bg_diff <= D_GATE: continue

            # --- GATE 2: ORE MATCH ---
            best_o, best_label = 0, ""
            for t in all_ore_t:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                if res.max() > best_o:
                    best_o, best_label = res.max(), t['name']

            # --- GATE 3: UI GHOST REJECTION (CYAN) ---
            if slot < 6:
                # Use standard template match for UI text
                best_u = max([cv2.matchTemplate(roi, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_t] + [0])
                
                # If match is weak OR we see the high intensity of white text
                if best_o < 0.85 and (best_u > U_GATE or np.max(roi[5:15, :]) > 240):
                    cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 255, 0), 1)
                    continue

            # --- GATE 4: FINAL ORE VERDICT ---
            # Added a mandatory background delta to prove it's an ore, not noise
            bg_match = max([cv2.matchTemplate(roi, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_t])
            
            if best_o > O_GATE and (best_o - bg_match > DELTA_GATE):
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                cv2.putText(raw_img, f"{best_label} ({best_o:.2f})", (x1+3, y1+44), 0, 0.3, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Purge_F{f_num}.jpg"), raw_img)

if __name__ == "__main__":
    run_ghost_purge_audit()