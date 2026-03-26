import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- TARGET WINDOWS ---
SHADOW_WINDOW = (20, 30) # For Index 26 Shadow Blindness
DURATION_WINDOW = (0, 1000) # To find minimum frames per floor
BUFFER_ROOT = "capture_buffer_0"

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic']
ORE_RESTRICTIONS = {'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25)}

# --- HELPERS ---

def get_slot_scores(roi_gray, mask, templates):
    # Raw correlation with background
    bg_scores = [cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']]
    bg_max = max(bg_scores) if bg_scores else 0
    
    # Raw correlation with blocks
    block_scores = [cv2.matchTemplate(roi_gray, block, cv2.TM_CCORR_NORMED, mask=mask).max() for block in templates['active']]
    block_max = max(ore_scores) if block_scores else 0
    
    return bg_max, block_max

# --- FORENSIC ENGINE ---

def run_v5_33_shadow_duration_audit():
    if not os.path.exists(BUFFER_ROOT):
        print(f"Error: {BUFFER_ROOT} not found.")
        return

    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # Load Templates (Strict Mac-safe Filter)
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

    # Use basic tiers for early diagnostic
    active_list = []
    for tier in ['dirt1', 'com1', 'rare1']:
        if tier in raw_tpls['block']: active_list.extend(raw_tpls['block'][tier])
    templates = {'active': active_list, 'bg': raw_tpls['bg']}
    
    mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)

    print(f"--- Forensic v5.33: Shadow Analysis (Indices {SHADOW_WINDOW[0]}-{SHADOW_WINDOW[1]}) ---")
    print(f"{'Idx':<6} | {'Slot':<5} | {'BG Score':<10} | {'Block Score':<10} | {'Delta':<6} | {'Status'}")
    print("-" * 65)

    for i in range(SHADOW_WINDOW[0], SHADOW_WINDOW[1] + 1):
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        # Check all 6 slots in Row 1
        for c in range(6):
            x1, y1 = int(SLOT1_CENTER[0] + (c * STEP_X)) - 24, int(SLOT1_CENTER[1]) - 24
            roi = img_gray[y1:y1+48, x1:x1+48]
            bg, block = get_slot_scores(roi, mask, templates)
            delta = block - bg
            
            # Labeling for your visual confirmation
            status = "Shadow?" if (bg > 0.85 and delta < 0.04 and delta > 0.01) else ""
            if i == 26 and c in [1, 5]: status = "!!! TARGET SHADOW !!!"
            
            print(f"{i:<6} | {c:<5} | {bg:<10.4f} | {ore:<10.4f} | {delta:<6.4f} | {status}")

    print(f"\n--- Duration Audit: Finding Minimum Frame Gaps ---")
    last_trigger_idx = 0
    gaps = []
    for i in range(DURATION_WINDOW[0], DURATION_WINDOW[1]):
        # Simulate a HUD shift (placeholder for speed)
        # We just want to see how close triggers naturally occur
        pass # (Data will be collected from your existing run logs)

    print("\n[PLAN] Use the BG/Block scores from Index 26 to set a strict 'Absolute Zero' for Vacuum.")

if __name__ == "__main__":
    run_v5_33_shadow_duration_audit()