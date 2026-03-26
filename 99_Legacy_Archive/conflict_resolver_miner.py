import cv2
import numpy as np
import os
import json
import csv

# --- SETTINGS FOR THE TEST ---
TARGET_RUN = "0"
# We'll scan a range of floors to see the effect
FLOOR_RANGE = range(10, 30) 
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def run_strict_miner_test():
    # 1. Load Templates (Block & BG)
    templates = {'block': {}, 'bg': []}
    t_path = "templates"
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"):
            templates['bg'].append(img)
        else:
            parts = f.split("_")
            tier, state = parts[0], parts[1]
            if tier not in templates['block']: templates['block'][tier] = {'act': [], 'sha': []}
            templates['block'][tier][state].append(img)

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running Strict Miner Test (Run {TARGET_RUN}) ---")

    for f_num in FLOOR_RANGE:
        if f_num not in sequence: continue
        img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0] + (col * STEP_X)), int(SLOT1_CENTER[1] + (row * STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi = gray[y1:y2, x1:x2]

            # 1. Check Background First
            bg_scores = [cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max() for t in templates['bg']]
            best_bg = max(bg_scores) if bg_scores else 0

            # 2. Check Blocks
            best_block = {'tier': 'empty', 'score': 0.0, 'state': 'none'}
            for tier, states in templates['block'].items():
                for state, imgs in states.items():
                    for t_img in imgs:
                        score = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED).max()
                        if state == 'sha': score *= 1.03 # The Shadow Boost
                        if score > best_block['score']:
                            best_block = {'tier': tier, 'score': score, 'state': state}

            # 3. NEW STRICT GATE
            # A block only 'wins' if:
            # - It beats the threshold (0.80)
            # - AND it beats the background by a significant gap (0.06)
            delta = best_block['score'] - best_bg
            
            if best_block['score'] > 0.80 and delta > 0.06:
                color = (0, 255, 0) if best_block['state'] == 'act' else (0, 165, 255)
                label = f"{best_block['tier']}"
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, label, (x1+2, y2-4), 0, 0.35, color, 1)

        cv2.imwrite(f"Test_Strict_F{f_num}.jpg", img)
        print(f"  Processed F{f_num}")

if __name__ == "__main__":
    run_strict_miner_test()