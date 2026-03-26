import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- TARGET WINDOW CONFIGURATION ---
START_IDX = 2020
BASE_IDX = 2021  # The frame we use as the Floor 27 "Baseline"
END_IDX = 2100
BUFFER_ROOT = "capture_buffer_0"

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

# --- LOGIC HELPERS ---

def format_dna(dna_str):
    return "|".join([dna_str[i:i+6] for i in range(0, 24, 6)])

def get_hamming_distance(dna1, dna2):
    return sum(c1 != c2 for c1, c2 in zip(dna1, dna2))

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
        if s < 0.85: s = max(s, cv2.matchTemplate(roi_gray, t_list[1], cv2.TM_CCORR_NORMED, mask=mask).max())
        if s > block_s: block_s = s

    delta = block_s - bg_s
    # Using v5.27 Rev1 thresholds for the diagnostic
    is_dirty = (delta > 0.085 if is_row1 else delta > 0.06) or (bg_s < 0.85)
    return "1" if is_dirty else "0"

# --- FORENSIC ENGINE ---

def run_hamming_diagnostic():
    if not os.path.exists(BUFFER_ROOT):
        print(f"Error: Path '{BUFFER_ROOT}' not found.")
        return

    # Load Templates for Floor 27
    raw_tpls = {'block': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f and any(f.startswith(tier) for tier in KNOWN_TIERS):
            parts = f.replace(".png", "").split("_")
            tier, state = parts[0], parts[1]
            if tier not in raw_tpls['block']: raw_tpls['block'][tier] = {'act': [], 'sha': []}
            m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
            raw_tpls['block'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])

    # Allowed Tiers for Floor 27
    active_list = []
    allowed = [t for t, (low, high) in ORE_RESTRICTIONS.items() if low <= 27 <= high]
    for tier in allowed:
        if tier in raw_tpls['block']:
            for state in ['act', 'sha']: active_list.extend(raw_tpls['block'][tier][state])
    runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}

    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)
    executor = ThreadPoolExecutor(max_workers=24)

    def get_dna(idx):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        tasks = []
        for c in range(24):
            r, col = divmod(c, 6)
            x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
            tasks.append((img_gray[y1:y1+48, x1:x1+48], img_bgr[y1:y1+48, x1:x1+48], 
                         text_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
        return "".join(list(executor.map(get_slot_status_worker, tasks)))

    # --- BASELINE ---
    print(f"Analyzing Frame {BASE_IDX} (Floor 27 Baseline)...")
    base_dna = get_dna(BASE_IDX)
    base_hud = cv2.imread(os.path.join(BUFFER_ROOT, files[BASE_IDX]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    
    print(f"\n{'Idx':<6} | {'HUD D.':<7} | {'Hamming':<7} | {'Row 1':<8} | {'DNA Profile'}")
    print("-" * 80)

    for i in range(START_IDX, END_IDX + 1):
        # HUD Check
        curr_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        curr_hud = curr_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(base_hud, curr_hud))

        # DNA Check
        curr_dna = get_dna(i)
        hamming = get_hamming_distance(base_dna, curr_dna)
        row1 = curr_dna[:6]

        note = ""
        if i == 2049: note = "<< F28 False Call"
        if i == 2074: note = "<< F29 False Call"

        print(f"{i:<6} | {hud_diff:<7.2f} | {hamming:<7} | {row1:<8} | {format_dna(curr_dna)} {note}")

    executor.shutdown()

if __name__ == "__main__":
    run_hamming_diagnostic()