import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- TARGET CONFIGURATION ---
START_IDX = 2020
BASE_IDX = 2021  # The frame we use as the "Truth" for Floor 27
END_IDX = 2100
BUFFER_ROOT = cfg.get_buffer_path(0)

# --- GAME CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# --- DNA LOGIC HELPERS ---

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[15:34, :] = 0 
    return mask

def is_xhair_present(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    return cv2.countNonZero(mask) > 8

def get_slot_status_worker(args):
    roi_gray, roi_bgr, mask, templates, is_row1 = args
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    if is_xhair_present(roi_bgr): return "1"

    block_s = 0.0
    for t_list in templates['active']:
        s = cv2.matchTemplate(roi_gray, t_list[0], cv2.TM_CCORR_NORMED, mask=mask).max()
        if not is_row1 and s < 0.85:
            s = max(s, cv2.matchTemplate(roi_gray, t_list[1], cv2.TM_CCORR_NORMED, mask=mask).max())
        if s > block_s: block_s = s

    delta = block_s - bg_s
    is_dirty = (delta > 0.065 if is_row1 else delta > 0.04) or (bg_s < 0.82)
    return "1" if is_dirty else "0"

# --- FORENSIC ENGINE ---

def run_forensic_dna_audit():
    if not os.path.exists(BUFFER_ROOT):
        print(f"Error: Could not find buffer directory '{BUFFER_ROOT}'")
        return

    # 1. Load Templates (Biome-filtered for Floor 27)
    raw_tpls = {'block': {}, 'bg': []}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f and any(f.startswith(tier) for tier in KNOWN_TIERS):
            parts = f.replace(".png", "").split("_")
            tier, state = parts[0], parts[1]
            if tier not in raw_tpls['block']: raw_tpls['block'][tier] = {'act': [], 'sha': []}
            m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
            raw_tpls['block'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])

    # Filter for Biome (Floor 27 usually uses Dirt2/Com2/Rare2)
    active_list = []
    for tier in ['dirt1', 'com1', 'rare1', 'epic1', 'leg1', 'dirt2', 'com2', 'rare2', 'epic2']:
        if tier in raw_tpls['block']:
            for state in ['act', 'sha']: active_list.extend(raw_tpls['block'][tier][state])
    runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}

    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)
    executor = ThreadPoolExecutor(max_workers=24)

    def calculate_dna(idx):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        tasks = []
        for c in range(24):
            r, col = divmod(c, 6)
            x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
            tasks.append((img_gray[y1:y1+48, x1:x1+48], img_bgr[y1:y1+48, x1:x1+48], 
                         text_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
        return "".join(list(executor.map(get_slot_status_worker, tasks)))

    # --- EXECUTION ---
    print(f"--- Forensic DNA Audit: Indices {START_IDX} - {END_IDX} ---")
    print("Gathering Ground Truth from Frame 2021...")
    truth_dna = calculate_dna(BASE_IDX)
    print(f"Base DNA (2021): {truth_dna}\n")
    
    print(f"{'Idx':<6} | {'HUD D.':<7} | {'DNA String':<25} | {'Flipped Bits'}")
    print("-" * 75)

    base_hud = cv2.imread(os.path.join(BUFFER_ROOT, files[BASE_IDX]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]

    for i in range(START_IDX, END_IDX + 1):
        # 1. HUD Check
        curr_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        curr_hud = curr_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(base_hud, curr_hud))

        # 2. DNA Check
        current_dna = calculate_dna(i)
        
        # 3. Bit Comparison
        flipped = [str(bit_idx) for bit_idx in range(24) if current_dna[bit_idx] != truth_dna[bit_idx]]
        flipped_str = ",".join(flipped) if flipped else "-"
        
        note = ""
        if i == 2049: note = "!! F28 FALSE CALL !!"
        if i == 2074: note = "!! F29 FALSE CALL !!"

        print(f"{i:<6} | {hud_diff:<7.2f} | {current_dna:<25} | {flipped_str:<12} {note}")

    executor.shutdown()

if __name__ == "__main__":
    run_forensic_dna_audit()