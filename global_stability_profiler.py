import cv2
import numpy as np
import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
REPORT_NAME = "global_stability_report.csv"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# --- GROUND TRUTH (Preserved for Reference/Future Use) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {8: 'myth1', 9: 'myth1', 14: 'myth1', 15: 'myth1'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {20: 'div1', 21: 'div1'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {5: "div2", 11: "div2", 17: "div2", 23: "div2"}}}
ORE_RESTRICTIONS = {'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74), 'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99), 'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)}

# --- HELPERS ---

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

# --- GLOBAL SCANNER ---

def run_v5_37_nitro():
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # Load Representative Templates for Global Awareness
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f and any(f.startswith(tier) for tier in KNOWN_TIERS):
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            if len(raw_tpls['ore'][tier]) < 3: raw_tpls['ore'][tier].append(img) # Keep list lean but global

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

        print(f"--- Launching v5.37-NITRO Global Scan ({len(files)} frames) ---")

        for i in range(len(files)):
            # SINGLE READ
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
                
                # SURGICAL GATE: Only run math if this specific slot moved
                if np.mean(cv2.absdiff(roi_now, roi_prev)) < 0.5:
                    curr_deltas[c] = last_deltas[c]
                else:
                    tasks.append((roi_now, txt_mask if c in [2,3] else std_mask, templates))
                    task_map.append(c)

            if tasks:
                results = list(executor.map(slot_worker, tasks))
                for idx, res in zip(task_map, results):
                    curr_deltas[idx] = res

            curr_bits = [1 if d > 0 else 0 for d in curr_deltas]
            hamming = sum(b1 != b2 for b1, b2 in zip(last_bits, curr_bits))
            
            is_pristine = all(b == 0 for b in curr_bits[:6])
            pristine_count = (pristine_count + 1) if is_pristine else 0
            
            writer.writerow([i, round(hud_diff, 4), hamming, "".join(map(str, curr_bits[:6])), pristine_count, round(np.mean(curr_deltas[:6]), 4)])
            
            # Update state
            last_bits, last_deltas, last_gray = curr_bits, curr_deltas, img_gray.copy()
            if hud_diff > 10.0: 
                last_hud = curr_hud.copy()
                csvfile.flush()

            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f" [PROGRESS] {i:05} | {elapsed:.1f}s | {i/elapsed:.1f} fps")

    executor.shutdown()
    print(f"\n[FINISH] Scanned {len(files)} in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v5_37_nitro()