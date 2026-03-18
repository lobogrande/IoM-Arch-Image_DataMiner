import pandas as pd
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
CSV_PATH = "global_stability_report.csv"
BUFFER_ROOT = "capture_buffer_0"
CHECK_WINDOW = (2450, 2550) # The area where we saw "Machine Gun" triggers
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# --- HELPERS (Same as Nitro to ensure parity) ---
def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:20, :] = 0 
    return mask

def slot_worker(args):
    roi, mask, templates = args
    bg_m = [cv2.matchTemplate(roi, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']]
    ore_m = [cv2.matchTemplate(roi, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['active']]
    return "1" if (max(ore_m) - max(bg_m)) > 0 else "0"

def run_integrity_audit():
    # 1. Load the "Nitro" Output
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 2. Load Templates (Exactly as Nitro did)
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            if len(raw_tpls['ore'][tier]) < 3: raw_tpls['ore'][tier].append(img)
    
    active_list = []
    for t_list in raw_tpls['ore'].values(): active_list.extend(t_list)
    templates = {'active': active_list, 'bg': raw_tpls['bg']}
    
    executor = ThreadPoolExecutor(max_workers=24)
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])

    print(f"--- Integrity Audit: Checking Frames {CHECK_WINDOW[0]}-{CHECK_WINDOW[1]} ---")
    print(f"{'Idx':<6} | {'Nitro Row1':<10} | {'Brute Row1':<10} | {'Status'}")
    print("-" * 50)

    mismatches = 0
    for i in range(CHECK_WINDOW[0], CHECK_WINDOW[1] + 1):
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        
        # Brute Force Scan (No Gating)
        tasks = []
        for c in range(6): # Just check Row 1 for speed
            x1 = int(SLOT1_CENTER[0] + (c * STEP_X)) - 24
            y1 = int(SLOT1_CENTER[1]) - 24
            roi = img_gray[y1:y1+48, x1:x1+48]
            tasks.append((roi, txt_mask if c in [2,3] else std_mask, templates))
        
        brute_bits = "".join(list(executor.map(slot_worker, tasks)))
        nitro_bits = str(df.iloc[i]['r1_bits']).zfill(6) # Ensure leading zeros

        status = "MATCH"
        if brute_bits != nitro_bits:
            status = "!!! MISMATCH !!!"
            mismatches += 1
        
        print(f"{i:<6} | {nitro_bits:<10} | {brute_bits:<10} | {status}")

    print(f"\nAudit Complete. Total Mismatches: {mismatches}")
    if mismatches == 0:
        print("RESULT: Nitro Gating is 100% loss-less for this window.")
    else:
        print("RESULT: Gating threshold (0.5) may be too high.")

if __name__ == "__main__":
    run_integrity_audit()