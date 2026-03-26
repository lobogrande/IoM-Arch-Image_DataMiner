import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- TARGET FRAMES (Ground Truth Provided by User) ---
# We are testing if the script can "see" these signatures through the text
TARGET_FRAMES = {
    43: "001000", # F2 Start (Ore in Slot 2, player in Slot 1)
    53: "000010", # F3 Start (Ore in Slot 4, player in Slot 3)
    63: "101010", # F4 Start (Ores in 0, 2, 4)
    82: "101010", # F5 Start
    92: "100010"  # F6 Start
}

BUFFER_ROOT = cfg.get_buffer_path(0)
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48

# --- DATA MAPPINGS ---
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']
# cfg.ORE_RESTRICTIONS moved to project_config

# --- HELPERS ---

def get_combined_mask(is_text_heavy_slot=False):
    """
    Creates the circular mask. 
    For Slots 2 & 3, we black out the top portion where 'Dig Stage' text sits.
    """
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot:
        # Aggressively mask out the top half of the circle where text overlaps
        mask[0:20, :] = 0 
    return mask

def get_slot_status_debug(roi_gray, mask, templates, delta_thresh):
    # Background match
    bg_matches = [cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']]
    bg_s = max(bg_matches) if bg_matches else 0
    
    # Ore match
    ore_s = 0.0
    for t_img in templates['active']:
        # Ensure template matches mask size (48x48)
        s = cv2.matchTemplate(roi_gray, t_img, cv2.TM_CCORR_NORMED, mask=mask).max()
        if s > ore_s: ore_s = s

    delta = ore_s - bg_s
    # If the ore is significantly stronger than background, or background is very weak
    return "1" if (delta > delta_thresh or bg_s < 0.80) else "0"

# --- FORENSIC ENGINE ---

def run_row1_transparency_audit():
    if not os.path.exists(BUFFER_ROOT):
        print(f"Error: {BUFFER_ROOT} not found.")
        return

    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    # 1. LOAD TEMPLATES (With Mac-file protection)
    raw_tpls = {'ore': {}, 'bg': []}
    if not os.path.exists(cfg.TEMPLATE_DIR):
        print("Error: 'templates' folder not found.")
        return

    for f in os.listdir(cfg.TEMPLATE_DIR):
        if f.startswith('.') or not f.lower().endswith('.png'): continue # Ignore .DS_Store etc.
        
        img = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, f), 0)
        if img is None: continue
        img = cv2.resize(img, (AI_DIM, AI_DIM))
        
        if f.startswith("background"):
            raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier in KNOWN_TIERS or any(tier.startswith(t) for t in KNOWN_TIERS):
                if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
                raw_tpls['ore'][tier].append(img)

    print(f"--- Row-1 Transparency Audit (Text-Overlap Focus) ---")
    print(f"{'Idx':<6} | {'Truth':<10} | {'D=0.04':<10} | {'D=0.06':<10} | {'D=0.08'}")
    print("-" * 62)

    std_mask = get_combined_mask(False)
    txt_mask = get_combined_mask(True)

    for idx, truth in TARGET_FRAMES.items():
        img_gray = cv2.imread(os.path.join(BUFFER_ROOT, files[idx]), 0)
        
        # Build templates for Floor 1-6 range
        active_list = []
        for tier in ['dirt1', 'com1', 'rare1']:
            if tier in raw_tpls['ore']:
                active_list.extend(raw_tpls['ore'][tier])
        
        runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}

        results = []
        for d_val in [0.04, 0.06, 0.08]:
            bits = []
            for c in range(6): 
                # Calculate ROI coordinates
                x1 = int(SLOT1_CENTER[0] + (c * STEP_X)) - 24
                y1 = int(SLOT1_CENTER[1]) - 24
                roi = img_gray[y1:y1+48, x1:x1+48]
                
                # Use text mask for slots 2 and 3
                current_mask = txt_mask if c in [2, 3] else std_mask
                
                bit = get_slot_status_debug(roi, current_mask, runtime_tpls, d_val)
                bits.append(bit)
            results.append("".join(bits))

        print(f"{idx:<6} | {truth:<10} | {results[0]:<10} | {results[1]:<10} | {results[2]}")

if __name__ == "__main__":
    run_row1_transparency_audit()