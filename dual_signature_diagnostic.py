import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 11)
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

# THE TRIPLE-GATE CALIBRATION
D_GATE = 6      # Heatmap difference threshold
C_GATE = 200    # Structural complexity (Laplacian Variance)
O_GATE = 0.72   # Lowered to catch Dirt1 now that gates provide safety

def get_ui_text_mask(slot_id):
    mask = np.zeros((48, 48), dtype=np.uint8)
    if slot_id in [1, 2, 3, 4]:
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def run_signature_peak_audit():
    # 1. Load Backgrounds (Stage 1)
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    
    # 2. Load Ores (Stage 3)
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v1.7 Signature Peak Auditor (Triple-Gate Mode) ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi = gray[y1:y2, x1:x2]

            # --- GATE 1: HEATMAP DIFFERENCE ---
            min_diff = min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_templates])
            if min_diff <= D_GATE: continue

            # --- GATE 2: STRUCTURAL COMPLEXITY ---
            # Player character is smooth/large; Ores are textured/jagged
            complexity = cv2.Laplacian(roi, cv2.CV_64F).var()
            if complexity < C_GATE: continue

            # --- GATE 3: MASKED IDENTIFICATION ---
            best_o = 0
            slot_mask = get_ui_text_mask(slot)
            for t in ore_templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=slot_mask)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_o: best_o = score
            
            # --- FINAL VERDICT ---
            if best_o > O_GATE:
                color = (0, 255, 0)
                label = f"O:{best_o:.2f}"
                cv2.rectangle(raw_img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (0,0,0), 2)
                cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"Peak_F{f_num}.jpg", raw_img)
        print(f" [+] Exported Floor {f_num}")

if __name__ == "__main__":
    run_signature_peak_audit()