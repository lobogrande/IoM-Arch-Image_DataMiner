import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import pandas as pd
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
CSV_PATH = "global_stability_report_v2.csv"
BUFFER_ROOT = cfg.get_buffer_path(0)
CHECK_WINDOW = (2450, 2550) 
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:20, :] = 0 
    return mask

def slot_worker(args):
    roi, mask, templates = args
    bg_m = [cv2.matchTemplate(roi, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']]
    block_m = [cv2.matchTemplate(roi, block, cv2.TM_CCORR_NORMED, mask=mask).max() for block in templates['active']]
    # Matching the logic from v5.37-NITRO
    return "1" if (max(ore_m) - max(bg_m)) > 0 else "0"

def run_integrity_audit_v2():
    # 1. LOAD CSV WITH STRICT STRING TYPES
    print(f"Loading {CSV_PATH} with string-strict parsing...")
    # dtype=str prevents the .0 and leading zero drop
    df = pd.read_csv(CSV_PATH, dtype={'r1_bits': str})
    
    # 2. LOAD TEMPLATES
    raw_tpls = {'block': {}, 'bg': []}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.resize(cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0), (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier not in raw_tpls['block']: raw_tpls['block'][tier] = []
            if len(raw_tpls['block'][tier]) < 3: raw_tpls['block'][tier].append(img)
    
    active_list = []
    for t_list in raw_tpls['block'].values(): active_list.extend(t_list)
    templates = {'active': active_list, 'bg': raw_tpls['bg']}
    
    executor = ThreadPoolExecutor(max_workers=24)
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])

    print(f"--- Corrected Integrity Audit: Frames {CHECK_WINDOW[0]}-{CHECK_WINDOW[1]} ---")
    print(f"{'Idx':<6} | {'Nitro Row1':<10} | {'Brute Row1':<10} | {'Status'}")
    print("-" * 50)

    logical_mismatches = 0
    for i in range(CHECK_WINDOW[0], CHECK_WINDOW[1] + 1):
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        
        # BRUTE: Re-calculate
        tasks = []
        for c in range(6):
            x1, y1 = int(SLOT1_CENTER[0] + (c * STEP_X)) - 24, int(SLOT1_CENTER[1]) - 24
            tasks.append((img_gray[y1:y1+48, x1:x1+48], txt_mask if c in [2,3] else std_mask, templates))
        brute_bits = "".join(list(executor.map(slot_worker, tasks)))

        # NITRO: Clean the data from the CSV
        raw_val = str(df.iloc[i]['r1_bits'])
        # Handle cases where Pandas already turned it into '101.0'
        nitro_bits = raw_val.split('.')[0].zfill(6)

        status = "MATCH"
        if brute_bits != nitro_bits:
            status = "!!! LOGIC MISMATCH !!!"
            logical_mismatches += 1
        
        print(f"{i:<6} | {nitro_bits:<10} | {brute_bits:<10} | {status}")

    print(f"\nAudit Complete. Total Logical Mismatches: {logical_mismatches}")
    if logical_mismatches == 0:
        print("RESULT: Nitro Gating is mathematically sound.")
    else:
        print(f"RESULT: {logical_mismatches} frames showed lag. Consider lowering gate to 0.3.")

if __name__ == "__main__":
    run_integrity_audit_v2()