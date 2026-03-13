import cv2
import numpy as np
import os
from datetime import datetime

# --- TARGETED CONFIG ---
TARGET_RUN = "0"
PROBLEM_FLOORS = [2, 5, 6, 14, 24, 37]
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
OUTPUT_DIR = f"diagnostic_results/Issue1_v38_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 6
O_GATE = 0.68
MIN_DELTA = 0.045 # Tightened delta

def has_ore_texture(roi):
    """
    Analyzes local variance. Text is 'flat' (solid white/gray).
    Ores are 'gritty' (high variance).
    """
    # Focus on the area where text usually causes ghosts (Top 40%)
    sample_zone = roi[5:22, 5:43]
    # Calculate Laplacian (Edge Density)
    laplacian = cv2.Laplacian(sample_zone, cv2.CV_64F).var()
    # UI Text usually results in a Laplacian Var < 100. 
    # Real Ores (Dirt/Stone) are usually > 450.
    return laplacian > 250

def run_issue1_v38_audit():
    # Assets
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    ui_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("negative_ui")]
    ore_t = []
    for f in os.listdir("templates"):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None: ore_t.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    # Load floor sequence
    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}

    print(f"--- Running Issue 1 Audit v3.8 (Entropy Validation) ---")

    for f_num in PROBLEM_FLOORS:
        if f_num not in seq: continue
        raw_img = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{seq[f_num]['frame']}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray[y1:y1+48, x1:x1+48]
            
            # 1. OCCUPANCY
            if min([np.sum(cv2.absdiff(roi, bg)) / (2304) for bg in bg_t]) <= D_GATE: continue

            # 2. COMPETITION
            mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6: cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
            else: cv2.circle(mask, (24, 24), 16, 255, -1)

            best_o, best_label = 0, ""
            for t in ore_t:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                if res.max() > best_o: best_o, best_label = res.max(), t['name']

            bg_match = max([cv2.matchTemplate(roi, bg, cv2.TM_CCOEFF_NORMED).max() for bg in bg_t])
            best_u = max([cv2.matchTemplate(roi, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_t] + [0])

            # 3. VERDICT WITH ENTROPY CHECK
            is_valid_ore = (best_o > O_GATE) and (best_o - bg_match > MIN_DELTA)
            
            if slot < 6:
                # If it fails texture test OR UI match is stronger than Ore, it's Cyan
                if not is_valid_ore or (best_u > best_o) or not has_ore_texture(roi):
                    cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 255, 0), 1)
                    continue

            if is_valid_ore:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                (w, h), _ = cv2.getTextSize(best_label, 0, 0.3, 1)
                cv2.rectangle(raw_img, (x1+2, y1+48-h-4), (x1+w+4, y1+48-2), (0,0,0), -1)
                cv2.putText(raw_img, best_label, (x1+3, y1+48-4), 0, 0.3, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"v38_F{f_num}.jpg"), raw_img)

if __name__ == "__main__":
    run_issue1_v38_audit()