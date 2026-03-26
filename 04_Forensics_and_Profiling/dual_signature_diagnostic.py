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
TARGET_FLOORS = range(1, 51)
BUFFER_ROOT = cfg.get_buffer_path(0)
TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
OUTPUT_DIR = f"diagnostic_results/Run_{TARGET_RUN}_v35_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GATES
D_GATE, O_GATE, P_GATE, U_GATE = 6, 0.68, 0.88, 0.82

def get_temporal_identity(coords, templates, frame_name, mask):
    files = sorted(os.listdir(BUFFER_ROOT))
    if frame_name not in files: return 0, ""
    idx = files.index(frame_name)
    best_s, best_n = 0, ""
    # Search +/- 15 frames for transient noise avoidance
    for off in [-15, -10, 10, 15]:
        if 0 <= idx+off < len(files):
            img = cv2.imread(os.path.join(BUFFER_ROOT, files[idx+off]), 0)
            if img is None: continue
            roi = img[coords[1]:coords[3], coords[0]:coords[2]]
            for t in templates:
                res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                if res.max() > best_s: best_s, best_n = res.max(), t['name']
    return best_s, best_n

def run_v35_audit():
    # 1. Load Assets
    bg_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("background")]
    player_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("negative_player")]
    ui_t = [cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48)) for f in os.listdir(cfg.TEMPLATE_DIR) if f.startswith("negative_ui")]
    all_ore_t = []
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if any(x in f for x in ["background", "negative"]): continue
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is not None: all_ore_t.append({'name': f.split("_")[0], 'img': cv2.resize(img, (48, 48))})

    with open(f"Unified_Consensus_Inputs/Run_{TARGET_RUN}/final_sequence.json", 'r') as f:
        seq = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v3.5 Verified Audit (F1-50) ---")

    for f_num in TARGET_FLOORS:
        if f_num not in seq: continue
        f_name = seq[f_num]['frame']
        raw_img = cv2.imread(os.path.join(f"Unified_Consensus_Inputs/Run_{TARGET_RUN}", f"F{f_num}_{f_name}"))
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        is_boss = f_num in cfg.BOSS_DATA
        # Filter templates by FLOOR RESTRICTION
        valid_ore_t = [t for t in all_ore_t if cfg.ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[0] <= f_num <= cfg.ORE_RESTRICTIONS.get(t['name'].lower(), (0,999))[1]]

        for slot in range(24):
            row, col = divmod(slot, 6)
            x1, y1 = int(74+(col*59.1))-24, int(261+(row*59.1))-24
            roi = gray[y1:y1+48, x1:x1+48]
            
            mask = np.zeros((48, 48), dtype=np.uint8)
            if slot < 6: cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
            else: cv2.circle(mask, (24, 24), 16, 255, -1)

            # --- GATE 1: OCCUPANCY (Skip for Boss) ---
            if not is_boss:
                if min([np.sum(cv2.absdiff(roi, bg)) / (48*48) for bg in bg_t]) <= D_GATE: continue

            # --- GATE 2: PLAYER REJECTION (PURPLE) (Skip for Boss) ---
            best_p = max([cv2.matchTemplate(roi, pt, cv2.TM_CCORR_NORMED).max() for pt in player_t] + [0])
            if not is_boss and best_p > P_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 0, 255), 1)
                continue

            # --- GATE 3: IDENTIFICATION ---
            best_o, best_label = 0, ""
            if is_boss:
                data = cfg.BOSS_DATA[f_num]
                best_label = (data['special'][slot] if data['tier'] == 'mixed' else data['tier'])
                best_o = 1.0 
            else:
                for t in valid_ore_t:
                    res = cv2.matchTemplate(roi, t['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    if res.max() > best_o: best_o, best_label = res.max(), t['name']

            # --- GATE 4: UI REJECTION (CYAN) (Skip for Boss) ---
            if not is_boss and slot < 6:
                # FIXED SYNTAX ERROR HERE
                best_u = max([cv2.matchTemplate(roi, ut, cv2.TM_CCORR_NORMED).max() for ut in ui_t] + [0])
                if best_o < 0.82 and (best_u > U_GATE or np.max(roi[5:15, :]) > 242):
                    cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (255, 255, 0), 1)
                    continue

            # --- TEMPORAL SCAN (FRAME SURFING) ---
            # Triggered if match is weak or crosshairs (high intensity) are detected
            if not is_boss and (best_o < 0.78 or (np.max(roi) > 235 and best_o < 0.90)):
                t_score, t_label = get_temporal_identity((x1, y1, x1+48, y1+48), valid_ore_t, f_name, mask)
                if t_score > best_o: best_o, best_label = t_score, t_label

            # --- FINAL RENDER ---
            if best_o > O_GATE:
                cv2.rectangle(raw_img, (x1, y1), (x1+48, y1+48), (0, 255, 0), 1)
                (w, h), _ = cv2.getTextSize(best_label, 0, 0.3, 1)
                cv2.rectangle(raw_img, (x1+2, y1+48-h-4), (x1+w+4, y1+48-2), (0,0,0), -1)
                cv2.putText(raw_img, best_label, (x1+3, y1+48-4), 0, 0.3, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"QA_F{f_num}.jpg"), raw_img)

if __name__ == "__main__":
    run_v35_audit()