import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
BUFFER_ROOT = cfg.get_buffer_path(0)
REPORT_NAME = "global_stability_report_v2.csv"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:20, :] = 0 
    return mask

def slot_worker(args):
    roi, mask, templates = args
    bg_s = max([cv2.matchTemplate(roi, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    ore_s = max([cv2.matchTemplate(roi, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['active']])
    return ore_s - bg_s

def run_v5_39_ultra():
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # Load Templates (Full Global Set)
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            if len(raw_tpls['ore'][tier]) < 4: raw_tpls['ore'][tier].append(img)

    active_list = []
    for t_list in raw_tpls['ore'].values(): active_list.extend(t_list)
    templates = {'active': active_list, 'bg': raw_tpls['bg']}

    executor = ThreadPoolExecutor(max_workers=24)
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)
    
    with open(REPORT_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'hud_diff', 'hamming', 'r1_bits', 'pristine_consec', 'avg_r1_delta'])

        last_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), 0)
        last_hud = last_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy()
        last_deltas = [0.0] * 24
        last_bits = [0] * 24
        pristine_count = 0
        start_time = time.time()

        print(f"--- Launching v5.39-ULTRA: Accuracy-First Scan ---")

        for i in range(len(files)):
            img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
            
            # HUD Check
            curr_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            hud_diff = np.mean(cv2.absdiff(last_hud, curr_hud))
            
            curr_deltas = [0.0] * 24
            tasks, task_map = [], []

            for c in range(24):
                r, col = divmod(c, 6)
                x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                roi_now = img_gray[y1:y1+48, x1:x1+48]
                roi_prev = last_gray[y1:y1+48, x1:x1+48]
                
                # REFINED GATE: 0.2 threshold + HUD Override
                if np.mean(cv2.absdiff(roi_now, roi_prev)) < 0.2 and hud_diff < 1.0:
                    curr_deltas[c] = last_deltas[c]
                else:
                    tasks.append((roi_now, txt_mask if c in [2,3] else std_mask, templates))
                    task_map.append(c)

            if tasks:
                results = list(executor.map(slot_worker, tasks))
                for idx, res in zip(task_map, results): curr_deltas[idx] = res

            curr_bits = [1 if d > 0 else 0 for d in curr_deltas]
            hamming = sum(b1 != b2 for b1, b2 in zip(last_bits, curr_bits))
            is_pristine = all(b == 0 for b in curr_bits[:6])
            pristine_count = (pristine_count + 1) if is_pristine else 0
            
            # Force bitstring to be 6 characters (avoids leading zero drop)
            r1_str = "".join(map(str, curr_bits[:6]))
            writer.writerow([i, round(hud_diff, 4), hamming, r1_str, pristine_count, round(np.mean(curr_deltas[:6]), 4)])
            
            last_bits, last_deltas, last_gray = curr_bits, curr_deltas, img_gray.copy()
            if hud_diff > 10.0: 
                last_hud = curr_hud.copy()
                csvfile.flush()

            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f" [PROGRESS] {i:05} | {elapsed:.1f}s | {i/elapsed:.1f} fps")

    executor.shutdown()
    print(f"\n[FINISH] Perfect Global Scan complete.")

if __name__ == "__main__":
    run_v5_39_ultra()