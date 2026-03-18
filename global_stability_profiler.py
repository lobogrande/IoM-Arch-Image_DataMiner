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

# --- HELPERS ---

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:20, :] = 0 
    return mask

def slot_worker(args):
    roi, mask, templates = args
    bg_m = [cv2.matchTemplate(roi, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']]
    bg_s = max(bg_m) if bg_m else 0
    ore_m = [cv2.matchTemplate(roi, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['active']]
    ore_s = max(ore_m) if ore_m else 0
    return ore_s - bg_s

# --- GLOBAL SCANNER ---

def run_v5_36_phantom():
    if not os.path.exists(BUFFER_ROOT):
        print(f"Error: {BUFFER_ROOT} not found.")
        return

    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # Load Templates
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f and any(f.startswith(tier) for tier in KNOWN_TIERS):
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            raw_tpls['ore'][tier].append(img)

    # Wide Awareness for Global Scan
    active_list = []
    for t in ['dirt1', 'com1', 'rare1', 'dirt2', 'com2', 'rare2']:
        if t in raw_tpls['ore']: active_list.extend(raw_tpls['ore'][t])
    templates = {'active': active_list, 'bg': raw_tpls['bg']}

    executor = ThreadPoolExecutor(max_workers=24)
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)
    
    with open(REPORT_NAME, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['idx', 'hud_diff', 'hamming', 'r1_bits', 'pristine_consec', 'avg_r1_delta'])

        # Initialize with a reduced-size "Low-Res" preview for the skip check
        last_preview = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        last_hud = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy()
        last_dna_bits = [0] * 24
        last_deltas = [0.0] * 24
        pristine_count = 0
        start_time = time.time()

        print(f"--- Launching v5.36-PHANTOM Global Scan ({len(files)} frames) ---")

        for i in range(len(files)):
            # 1. FAST PREVIEW: Load 1/16th resolution image for the skip decision
            curr_preview = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), cv2.IMREAD_REDUCED_GRAYSCALE_4)
            pixel_diff = np.mean(cv2.absdiff(curr_preview, last_preview))
            
            # 2. DECISION LOGIC
            # If the board is basically static, just record the previous state
            if pixel_diff < 1.0:
                hud_diff = 0.0
                hamming = 0
                curr_bits = last_dna_bits
                deltas = last_deltas
            else:
                # Board is active, load full-res and do the math
                img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
                curr_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
                hud_diff = np.mean(cv2.absdiff(last_hud, curr_hud))
                
                tasks = []
                for c in range(24):
                    r, col = divmod(c, 6)
                    x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                    roi = img_gray[y1:y1+48, x1:x1+48]
                    tasks.append((roi, txt_mask if c in [2,3] else std_mask, templates))
                
                deltas = list(executor.map(slot_worker, tasks))
                curr_bits = [1 if d > 0 else 0 for d in deltas]
                hamming = sum(b1 != b2 for b1, b2 in zip(last_dna_bits, curr_bits))
                
                if hud_diff > 10.0:
                    last_hud = curr_hud.copy()
                    csvfile.flush()

            is_pristine = all(b == 0 for b in curr_bits[:6])
            pristine_count = (pristine_count + 1) if is_pristine else 0
            
            r1_bits_str = "".join(map(str, curr_bits[:6]))
            avg_r1_delta = np.mean(deltas[:6])
            
            writer.writerow([i, round(hud_diff, 4), hamming, r1_bits_str, pristine_count, round(avg_r1_delta, 4)])
            
            last_dna_bits = curr_bits
            last_deltas = deltas
            last_preview = curr_preview.copy()

            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f" [PROGRESS] {i:05} | {elapsed:.1f}s | {i/elapsed:.1f} fps")

    executor.shutdown()
    print(f"\n[FINISH] Scanned {len(files)} in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    run_v5_36_phantom()