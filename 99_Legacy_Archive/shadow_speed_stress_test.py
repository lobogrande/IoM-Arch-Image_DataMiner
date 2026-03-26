import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# --- TARGET WINDOWS ---
WINDOWS = [(20, 50), (2450, 2500)] 
BUFFER_ROOT = "capture_buffer_0"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48

# --- DATA MAPPINGS ---
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg']
ORE_RESTRICTIONS = {'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (1, 25), 'dirt2': (1, 30)}

# --- HELPERS ---

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:20, :] = 0 
    return mask

def get_slot_scores(roi_gray, mask, templates):
    bg_matches = [cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']]
    bg_max = max(bg_matches) if bg_matches else 0
    block_matches = [cv2.matchTemplate(roi_gray, block, cv2.TM_CCORR_NORMED, mask=mask).max() for block in templates['active']]
    block_max = max(ore_matches) if block_matches else 0
    return bg_max, block_max

# --- FORENSIC ENGINE ---

def run_v5_34_stress_test():
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # Load Templates (Strict Mac-safe Filter)
    raw_tpls = {'block': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier not in raw_tpls['block']: raw_tpls['block'][tier] = []
            raw_tpls['block'][tier].append(img)

    active_list = []
    for t in ['dirt1', 'com1', 'rare1', 'dirt2']:
        if t in raw_tpls['block']: active_list.extend(raw_tpls['block'][t])
    templates = {'active': active_list, 'bg': raw_tpls['bg']}
    
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)

    print(f"--- Forensic v5.34: Delta-Zero & Velocity Audit ---")
    
    for start, end in WINDOWS:
        print(f"\n[SCANNING WINDOW: {start}-{end}]")
        print(f"{'Idx':<6} | {'Row 1 Deltas (0-5)':<40} | {'Pristine?'}")
        print("-" * 70)
        
        for i in range(start, end + 1):
            img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
            deltas = []
            is_pristine = True
            
            for c in range(6): 
                x1, y1 = int(SLOT1_CENTER[0] + (c * STEP_X)) - 24, int(SLOT1_CENTER[1]) - 24
                roi = img_gray[y1:y1+48, x1:x1+48]
                bg, block = get_slot_scores(roi, txt_mask if c in [2,3] else std_mask, templates)
                delta = block - bg
                deltas.append(f"{delta:+.3f}")
                if delta > 0: is_pristine = False # The "All-or-Nothing" Rule
            
            p_status = "YES" if is_pristine else "NO (Block/Shadow detected)"
            print(f"{i:<6} | {' '.join(deltas)} | {p_status}")

if __name__ == "__main__":
    run_v5_34_stress_test()