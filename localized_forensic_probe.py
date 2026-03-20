import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- DIAGNOSTIC TARGETS ---
TARGETS = [
    {"idx": 723,  "desc": "F17 Banner (Miss-call)"},
    {"idx": 1606, "desc": "F24 Banner (Miss-call)"},
    {"idx": 1630, "desc": "F25 Arrival (Late-call)"}
]

# --- v5.53 LOGIC CONSTANTS ---
ROI_OLD = (54, 74, 103, 138)
ROI_NEW = (58, 72, 104, 137) # Surgical ROI for comparison
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def get_slot_status_worker(args):
    roi, mask, templates, is_row1 = args
    bg_s = max([cv2.matchTemplate(roi, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    ore_s = max([cv2.matchTemplate(roi, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['ore_all']])
    delta = ore_s - bg_s
    thresh = 0.08 if is_row1 else 0.05
    return "1" if (delta > thresh or bg_s < 0.83) else "0"

def run_forensic_diagnostics():
    buffer_root = "capture_buffer_0"
    all_files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    # Load Templates
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            raw_tpls['ore'][tier].append(img)

    std_mask = np.zeros((48, 48), dtype=np.uint8)
    cv2.circle(std_mask, (24, 24), 18, 255, -1)
    txt_mask = std_mask.copy(); txt_mask[0:24, :] = 0 
    executor = ThreadPoolExecutor(max_workers=24)

    print(f"{'Idx':<6} | {'HUD Old':<7} | {'HUD New':<7} | {'Ham':<4} | {'R1':<3} | {'Pers':<4} | {'Logic Trigger'}")
    print("-" * 85)

    for target in TARGETS:
        t_idx = target['idx']
        print(f"\n--- Analyzing {target['desc']} ---")
        
        # We look 10 frames before and 5 frames after the event
        for i in range(t_idx - 10, t_idx + 6):
            if i <= 0: continue
            
            # Load current and previous (anchor) for diffs
            curr_img = cv2.imread(os.path.join(buffer_root, all_files[i]), 0)
            # For forensics, we'll assume the previous frame was the "anchor" for diffing
            prev_img = cv2.imread(os.path.join(buffer_root, all_files[i-1]), 0)
            
            # HUD Diffs
            hud_old = np.mean(cv2.absdiff(curr_img[ROI_OLD[0]:ROI_OLD[1], ROI_OLD[2]:ROI_OLD[3]], 
                                          prev_img[ROI_OLD[0]:ROI_OLD[1], ROI_OLD[2]:ROI_OLD[3]]))
            hud_new = np.mean(cv2.absdiff(curr_img[ROI_NEW[0]:ROI_NEW[1], ROI_NEW[2]:ROI_NEW[3]], 
                                          prev_img[ROI_NEW[0]:ROI_NEW[1], ROI_NEW[2]:ROI_NEW[3]]))
            
            # DNA Logic (Simplified for diagnostic)
            # Use all available ores for the probe
            all_ores = [img for t in raw_tpls['ore'] for img in raw_tpls['ore'][t]]
            tpls = {'ore_all': all_ores, 'bg': raw_tpls['bg']}
            
            dna_tasks = []
            for c in range(24):
                r, col = divmod(c, 6)
                x, y = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                dna_tasks.append((curr_img[y:y+48, x:x+48], txt_mask if c in [2,3] else std_mask, tpls, (c<6)))
            
            curr_dna = "".join(list(executor.map(get_slot_status_worker, dna_tasks)))
            
            # Anchor DNA check (for this probe, we compare to frame i-1)
            prev_dna_tasks = []
            for c in range(24):
                r, col = divmod(c, 6)
                x, y = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                prev_dna_tasks.append((prev_img[y:y+48, x:x+48], txt_mask if c in [2,3] else std_mask, tpls, (c<6)))
            prev_dna = "".join(list(executor.map(get_slot_status_worker, prev_dna_tasks)))
            
            hamming = sum(c1 != c2 for c1, c2 in zip(prev_dna, curr_dna))
            row1_count = curr_dna[:6].count("1")
            
            # Logic flags
            strike = (hud_old > 3.0 and hamming >= 4)
            vacuum = (row1_count <= 1 and hamming >= 5)
            trigger = "STRIKE" if strike else "VACUUM" if vacuum else "-"
            
            print(f"{i:05} | {hud_old:<7.2f} | {hud_new:<7.2f} | {hamming:<4} | {row1_count:<3} | {trigger}")

    executor.shutdown()

if __name__ == "__main__":
    run_forensic_diagnostics()