import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- TARGET WINDOW CONFIGURATION ---
START_IDX = 0
END_IDX = 300
BUFFER_ROOT = cfg.get_buffer_path(0)
HEADER_ROI = (54, 74, 103, 138)
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48

# --- GROUND TRUTH DATA (FULL & PERMANENT) ---
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# cfg.BOSS_DATA moved to project_config

# cfg.ORE_RESTRICTIONS moved to project_config

# --- HELPERS ---

def get_hamming_distance(dna1, dna2):
    return sum(c1 != c2 for c1, c2 in zip(dna1, dna2))

def format_dna(dna_str):
    return "|".join([dna_str[i:i+6] for i in range(0, 24, 6)])

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
    # Using v5.27 stabilized thresholds
    is_dirty = (delta > 0.085 if is_row1 else delta > 0.06) or (bg_s < 0.85)
    return "1" if is_dirty else "0"

# --- ENGINE ---

def run_transition_signature_audit():
    if not os.path.exists(BUFFER_ROOT):
        print(f"Error: Path '{BUFFER_ROOT}' not found.")
        return

    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # 1. LOAD ALL TEMPLATES WITH STRICT FILTER
    raw_tpls = {'block': {}, 'bg': []}
    for f in os.listdir(cfg.TEMPLATE_DIR):
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): 
            raw_tpls['bg'].append(img)
            continue
        
        # STRICT FILTER: Skip if not a known block tier
        if not any(f.startswith(tier) for tier in KNOWN_TIERS): continue
        if "_" not in f: continue

        parts = f.replace(".png", "").split("_")
        tier, state = parts[0], parts[1]
        if tier not in raw_tpls['block']: raw_tpls['block'][tier] = {'act': [], 'sha': []}
        if state in ['act', 'sha']:
            m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
            raw_tpls['block'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])

    executor = ThreadPoolExecutor(max_workers=24)
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)

    def get_dna_at(idx, current_floor):
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Determine allowed tiers for current floor
        allowed = [t for t, (low, high) in cfg.ORE_RESTRICTIONS.items() if low <= current_floor <= high]
        if current_floor in cfg.BOSS_DATA:
            bt = cfg.BOSS_DATA[current_floor]['tier']
            if bt != 'mixed' and bt not in allowed: allowed.append(bt)
            
        active_list = []
        for tier in allowed:
            if tier in raw_tpls['block']:
                for state in ['act', 'sha']: active_list.extend(raw_tpls['block'][tier][state])
        
        runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}
        tasks = []
        for c in range(24):
            r, col = divmod(c, 6)
            x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
            tasks.append((img_gray[y1:y1+48, x1:x1+48], img_bgr[y1:y1+48, x1:x1+48], text_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
        return "".join(list(executor.map(get_slot_status_worker, tasks)))

    print(f"--- Real Transition Signature Audit (Frames {START_IDX}-{END_IDX}) ---")
    
    # Establish Floor 1 Baseline
    anchor_dna = get_dna_at(0, 1)
    anchor_hud = cv2.imread(os.path.join(BUFFER_ROOT, files[0]), 0)[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
    
    print(f"{'Idx':<6} | {'HUD D.':<7} | {'Hamming':<7} | {'Row 1':<8} | {'DNA Profile'}")
    print("-" * 80)

    for i in range(START_IDX, END_IDX):
        curr_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        curr_hud = curr_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(anchor_hud, curr_hud))
        
        # We check DNA against the baseline to see the "Signature" of a change
        curr_dna = get_dna_at(i, 1) # Forced Floor 1 restriction for baseline stability
        hamming = get_hamming_distance(anchor_dna, curr_dna)
        row1 = curr_dna[:6]

        note = ""
        if i == 43: note = "<< F2 CALLED"
        if i == 63: note = "<< F3 MISSED"

        # Output significant frames
        if hamming > 0 or hud_diff > 1.0:
            print(f"{i:<6} | {hud_diff:<7.2f} | {hamming:<7} | {row1:<8} | {format_dna(curr_dna)} {note}")

    executor.shutdown()

if __name__ == "__main__":
    run_transition_signature_audit()