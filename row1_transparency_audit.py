import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# --- TARGET FRAMES (Ground Truth Provided by User) ---
TARGET_FRAMES = {
    43: "101100", # F2 Start (Adjusted based on your image analysis)
    53: "000010", # F3 Start
    63: "101010", # F4 Start
    82: "101010", # F5 Start
    92: "100010"  # F6 Start
}

BUFFER_ROOT = "capture_buffer_0"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48

# --- DATA MAPPINGS ---
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29)
}

# --- HELPERS ---

def get_combined_mask(is_text_heavy_slot=False, aggression_level=0):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot:
        # Testing different vertical cutoffs to see through "Dig Stage" text
        cutoff = 15 + (aggression_level * 2)
        mask[cutoff:34, :] = 0 
    return mask

def get_slot_status_debug(args):
    roi_gray, mask, templates, is_row1, delta_thresh = args
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    
    ore_s = 0.0
    for t_list in templates['active']:
        s = cv2.matchTemplate(roi_gray, t_list[0], cv2.TM_CCORR_NORMED, mask=mask).max()
        if s > ore_s: ore_s = s

    delta = ore_s - bg_s
    return "1" if (delta > delta_thresh or bg_s < 0.85) else "0"

# --- FORENSIC ENGINE ---

def run_row1_transparency_audit():
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None or "_" not in f: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        else:
            parts = f.replace(".png", "").split("_")
            tier, state = parts[0], parts[1]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = {'act': []}
            if state == 'act': raw_tpls['ore'][tier]['act'].append(img)

    print(f"--- Row-1 Transparency Audit (Aggression Testing) ---")
    print(f"{'Idx':<6} | {'Truth':<10} | {'D=0.04':<10} | {'D=0.06':<10} | {'D=0.08'}")
    print("-" * 60)

    for idx, truth in TARGET_FRAMES.items():
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]), 0)
        
        # Build templates for Floor 1-6 range
        active_list = []
        for tier in ['dirt1', 'com1', 'rare1']:
            if tier in raw_tpls['ore']: active_list.extend(raw_tpls['ore'][tier]['act'])
        runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}

        results = []
        for d_val in [0.04, 0.06, 0.08]:
            bits = []
            for c in range(6): # Only checking Row 1
                r, col = 0, c
                x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])-24
                roi = img_gray[y1:y1+48, x1:x1+48]
                mask = get_combined_mask(c in [2,3], aggression_level=1)
                bits.append(get_slot_status_debug((roi, mask, runtime_tpls, True, d_val)))
            results.append("".join(bits))

        print(f"{idx:<6} | {truth:<10} | {results[0]:<10} | {results[1]:<10} | {results[2]}")

if __name__ == "__main__":
    run_row1_transparency_audit()