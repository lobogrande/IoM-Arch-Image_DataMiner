import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- TARGET WINDOW CONFIGURATION ---
START_IDX = 2020
BASE_IDX = 2021  # Floor 27 Baseline
END_IDX = 2100
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "forensic_v531_output"

# --- GAME CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# --- DATA MAPPINGS ---
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- HELPERS ---

def get_hamming_distance(dna1, dna2):
    return sum(c1 != c2 for c1, c2 in zip(dna1, dna2))

def format_dna(dna_str):
    return "|".join([dna_str[i:i+6] for i in range(0, 24, 6)])

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot:
        mask[0:20, :] = 0 
    return mask

def get_slot_status_worker(args):
    roi_gray, mask, templates, is_row1 = args
    # Use the high-sensitivity 0.04 threshold found in v5.30
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    
    block_s = 0.0
    for t_img in templates['active']:
        s = cv2.matchTemplate(roi_gray, t_img, cv2.TM_CCORR_NORMED, mask=mask).max()
        if s > block_s: block_s = s

    delta = block_s - bg_s
    return "1" if (delta > 0.04 or bg_s < 0.85) else "0"

# --- FORENSIC ENGINE ---

def run_hamming_sensitivity_crosscheck():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # 1. LOAD TEMPLATES (Strict Filter)
    raw_tpls = {'block': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if any(tier.startswith(t) for t in KNOWN_TIERS):
                if tier not in raw_tpls['block']: raw_tpls['block'][tier] = []
                raw_tpls['block'][tier].append(img)

    executor = ThreadPoolExecutor(max_workers=24)
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)

    def get_dna(idx, floor_context):
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]), 0)
        allowed = [t for t, (low, high) in ORE_RESTRICTIONS.items() if low <= floor_context <= high]
        active_list = []
        for tier in allowed:
            if tier in raw_tpls['block']: active_list.extend(raw_tpls['block'][tier])
        
        runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}
        tasks = []
        for c in range(24):
            r, col = divmod(c, 6)
            x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
            tasks.append((img_gray[y1:y1+48, x1:x1+48], txt_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
        return "".join(list(executor.map(get_slot_status_worker, tasks)))

    # --- BASELINE ---
    print(f"Establishing Floor 27 Baseline (Frame {BASE_IDX})...")
    base_dna = get_dna(BASE_IDX, 27)
    base_hud = cv2.imread(os.path.join(BUFFER_ROOT, files[BASE_IDX]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    
    print(f"\n{'Idx':<6} | {'HUD D.':<7} | {'Hamming':<7} | {'Action'}")
    print("-" * 45)

    for i in range(START_IDX, END_IDX + 1):
        curr_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[i]))
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        curr_hud = curr_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(base_hud, curr_hud))

        curr_dna = get_dna(i, 27)
        hamming = get_hamming_distance(base_dna, curr_dna)

        action = ""
        # Simulate the 'New Floor' logic we intend to use
        # Requires HUD shift AND at least 5 bits changed
        if hud_diff > 5.0 and hamming >= 5:
            action = "DETECTED NEW FLOOR"
            cv2.imwrite(f"{OUT_DIR}/F_TRIGGER_Idx{i:05}_H{hamming}.jpg", curr_bgr)
        elif hud_diff > 5.0:
            action = "HUD ONLY (Rejected by Hamming)"
        
        if hud_diff > 1.0 or hamming > 0:
            print(f"{i:<6} | {hud_diff:<7.2f} | {hamming:<7} | {action}")

    executor.shutdown()
    print(f"\nAudit complete. Trigger images saved to '{OUT_DIR}'")

if __name__ == "__main__":
    run_hamming_sensitivity_crosscheck()