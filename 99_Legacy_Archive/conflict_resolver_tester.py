import cv2
import numpy as np
import os
import json

# --- TARGET CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 18 
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def run_conflict_resolver():
    # 1. Load Templates
    templates = {'ore': {}, 'bg': []}
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
            if tier not in templates['ore']: templates['ore'][tier] = {'act': [], 'sha': []}
            templates['ore'][tier][state].append(img)

    # 2. Load Image
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}
    img = cv2.imread(os.path.join(run_path, f"F{TARGET_FLOOR}_{sequence[TARGET_FLOOR]['frame']}"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"--- Conflict Resolver: Slot-by-Slot Scoreboard ---")

    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0] + (col * STEP_X)), int(SLOT1_CENTER[1] + (row * STEP_Y))
        x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
        roi = gray[y1:y2, x1:x2]

        # Calculate Best BG
        best_bg = max([cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max() for t in templates['bg']])
        
        # Calculate Best Ore
        best_ore_score = 0
        best_ore_name = "None"
        for tier, states in templates['ore'].items():
            for state, imgs in states.items():
                for t_img in imgs:
                    score = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED).max()
                    if state == 'sha': score *= 1.03 # The Shadow Boost
                    if score > best_ore_score:
                        best_ore_score = score
                        best_ore_name = tier

        # Draw Comparison
        color = (0, 255, 0) if best_ore_score > 0.77 and (best_ore_score > best_bg) else (255, 0, 0)
        label1 = f"O:{best_ore_name[:4]} {best_ore_score:.2f}"
        label2 = f"B:{best_bg:.2f}"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, label1, (x1+2, y1+12), 0, 0.3, color, 1)
        cv2.putText(img, label2, (x1+2, y2-4), 0, 0.3, (200, 200, 200), 1)

    cv2.imwrite("Conflict_Scoreboard.jpg", img)
    print("Generated Conflict_Scoreboard.jpg. Look for slots where 'O' is barely beating 'B'.")

if __name__ == "__main__":
    run_conflict_resolver()