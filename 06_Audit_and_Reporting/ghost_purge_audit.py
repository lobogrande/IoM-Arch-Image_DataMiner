import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- GROUND TRUTH DATA ---
# cfg.BOSS_DATA moved to project_config

# cfg.ORE_RESTRICTIONS moved to project_config

# --- CONFIG ---
TARGET_RUN = "0"
UNIFIED_ROOT = f"Unified_Consensus_Inputs/Run_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FullRun0_v41_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE = 6      
O_GATE = 0.68   
P_GATE = 0.88   # Sensitivity for player detection

def run_v41_audit():
    # 1. Load Assets
    bg_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    player_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("negative_player")]
    ui_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("negative_ui")]
    all_block_t = []
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is not None:
            all_block_t.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(os.path.join(UNIFIED_ROOT, "final_sequence.json"), 'r') as f:
        full_sequence = json.load(f)

    print(f"--- Running v4.1 Robust Full-Run Audit ---")

    for entry in full_sequence:
        f_num = entry['floor']
        f_name = entry['frame']
        raw_img = cv2.imread(os.path.join(UNIFIED_ROOT, f"F{f_num}_{f_name}"))
        if raw_img is None: continue
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        is_boss = f_num in cfg.BOSS_DATA
        valid_templates = [t for t in all_block_t if cfg.ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[0] <= f_num <= cfg.ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[1]]

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray[y1:y1+48, x1:x1+48]

            # --- STEP 1: PLAYER REJECTION (PRIORITY) ---
            # We run this on EVERY slot before masking to ensure the reaper doesn't hide in HUD zones
            best_p = max([cv2.matchTemplate(roi, pt, cv2.TM_CCORR_NORMED).max() for pt in player_t] + [0])
            if not is_boss and best_p > P_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 0, 255), 1)
                continue

            # --- STEP 2: OCCUPANCY ---
            if not is_boss and min([np.sum(cv2.absdiff(roi, bg)) / (2304) for bg in bg_t]) <= D_GATE:
                continue

            # --- STEP 3: IDENTIFICATION ---
            mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6: cv2.rectangle(mask, (5, 22), (43, 45), 255, -1)
            else: cv2.circle(mask, (24, 24), 16, 255, -1)

            best_o, best_label = 0, ""
            if is_boss:
                data = cfg.BOSS_DATA[f_num]
                best_label = (data['special'][slot] if data['tier'] == 'mixed' else data['tier'])
                best_o = 1.0
            else:
                for t in valid_templates:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    if res.max() > best_o: best_o, best_label = res.max(), t['name']

            # --- STEP 4: REFINED ZONAL GHOST REJECTION ---
            if not is_boss and slot < 6:
                best_u = max([cv2.matchTemplate(roi, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_t] + [0])
                bottom_roi = roi[24:48, :]
                bottom_bg_diff = min([np.sum(cv2.absdiff(bottom_roi, bg[24:48, :])) / (1152) for bg in bg_t])
                
                # REJECT IF:
                # - UI match strictly beats Block match
                # - Bottom half is TOO similar to background (Rescued threshold: 3.5)
                # - Peak white exists without elite confidence (>0.90)
                if (best_u > best_o) or (bottom_bg_diff < 3.5) or (np.max(roi[5:15, :]) > 242 and best_o < 0.90):
                    cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 255, 0), 1)
                    continue

            # --- STEP 5: FINAL BOXING ---
            if best_o > O_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                label_text = f"{best_label} ({best_o:.2f})"
                (tw, th), _ = cv2.getTextSize(label_text, 0, 0.3, 1)
                cv2.rectangle(raw_img, (x1+2, y1+48-th-4), (x1+tw+4, y1+48-2), (0,0,0), -1)
                cv2.putText(raw_img, label_text, (x1+3, y1+48-4), 0, 0.3, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"F{f_num}_v41Audit.jpg"), raw_img)

if __name__ == "__main__":
    run_v41_audit()