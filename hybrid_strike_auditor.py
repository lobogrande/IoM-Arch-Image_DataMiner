import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- FORENSIC CONTROL ---
START_INDEX = 0      # Start at frame X
END_INDEX = 25291    # End at frame Y (Set to None for full dataset)
OUT_DIR = "calibrated_audit_v545"

# --- CALIBRATED CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
PLAYER_ROI_Y = (120, 420)
ROW_0_MAX_Y = 300 
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# --- GROUND TRUTH DATA ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 
    34: {'tier': 'mixed', 'special': {8: 'myth1', 9: 'myth1', 14: 'myth1', 15: 'myth1'}}, 
    35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 
    49: {"tier": "mixed", "special": {18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}},
    74: {'tier': 'mixed', 'special': {20: 'div1', 21: 'div1'}}, 98: {'tier': 'myth3'}, 
    99: {"tier": "mixed", "special": {5: "div2", 11: "div2", 17: "div2", 23: "div2"}}
}

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- HELPERS ---

def get_hamming_distance(dna1, dna2):
    return sum(c1 != c2 for c1, c2 in zip(dna1, dna2))

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:32, :] = 0  # NEW: More aggressive (Lower 1/3)
    return mask

def is_player_in_slot0(roi_gray, p_right, p_left):
    slot0_roi = roi_gray[:, 0:110]
    res_r = cv2.matchTemplate(slot0_roi, p_right, cv2.TM_CCOEFF_NORMED)
    res_l = cv2.matchTemplate(slot0_roi, p_left, cv2.TM_CCOEFF_NORMED)
    return max(cv2.minMaxLoc(res_r)[1], cv2.minMaxLoc(res_l)[1]) > 0.65

def get_slot_status_worker(args):
    roi_gray, mask, templates, is_row1 = args
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    ore_s = max([cv2.matchTemplate(roi_gray, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['active']])
    delta = ore_s - bg_s
    
    # CALIBRATED: 0.08 for Row 1 (Text-Filter), 0.06 for others
    thresh = 0.08 if is_row1 else 0.06
    return "1" if (delta > thresh or bg_s < 0.82) else "0"

# --- MAIN ENGINE ---

def run_v5_45_calibrated_auditor():
    buffer_root = "capture_buffer_0"
    os.makedirs(f"{OUT_DIR}/confirmed", exist_ok=True)
    
    p_right, p_left = cv2.imread("templates/player_right.png", 0), cv2.imread("templates/player_left.png", 0)
    
    # 1. LOAD ALL TEMPLATES
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f and any(f.startswith(tier) for tier in KNOWN_TIERS):
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            raw_tpls['ore'][tier].append(img)

    all_files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    files = all_files[START_INDEX:END_INDEX] if END_INDEX else all_files[START_INDEX:]

    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)
    
    # Initialize Anchor
    first_img = cv2.imread(os.path.join(buffer_root, files[0]))
    last_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    anchor = {"idx": START_INDEX, "dna": "0"*24, "hud": last_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy()}
    
    dna_memory, confirmed_count = ["0"] * 24, 1
    lockout_timer = 0
    executor = ThreadPoolExecutor(max_workers=24)
    start_time = time.time()

    print(f"--- Running v5.45: Calibrated Hybrid Auditor (Range: {START_INDEX}-{END_INDEX if END_INDEX else 'EOF'}) ---")

    for i in range(1, len(files)):
        abs_idx = START_INDEX + i
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
        is_p_in_0 = is_player_in_slot0(img_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], :], p_right, p_left)

        # DYNAMIC HUD THRESHOLD: Fades from 6.0 (Early) to 4.5 (Late)
        dynamic_thresh = max(4.5, 6.0 - (confirmed_count / 110) * 1.5)

        if lockout_timer > 0: lockout_timer -= 1

        # TRIGGER GATE
        if hud_diff > dynamic_thresh or not is_p_in_0:
            allowed = [t for t, (low, high) in ORE_RESTRICTIONS.items() if low <= confirmed_count <= high]
            active_list = []
            for tier in allowed:
                if tier in raw_tpls['ore']: active_list.extend(raw_tpls['ore'][tier])
            runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}

            tasks, task_indices = [], []
            for c in range(24):
                r, col = divmod(c, 6)
                x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                
                roi_now, roi_prev = img_gray[y1:y1+48, x1:x1+48], last_gray[y1:y1+48, x1:x1+48]
                if np.mean(cv2.absdiff(roi_now, roi_prev)) < 0.5: continue 
                
                tasks.append((roi_now, txt_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
                task_indices.append(c)
            
            if tasks:
                results = list(executor.map(get_slot_status_worker, tasks))
                for idx, res in zip(task_indices, results): dna_memory[idx] = res
            
            this_dna = "".join(dna_memory)
            hamming = get_hamming_distance(anchor['dna'], this_dna)
            row1_count = dna_memory[:6].count("1")

            # PULSE VALIDATION: Only allow new floor if lockout is 0 and signal is stable
            if lockout_timer == 0:
                # VACUUM (Catching F3/F5/F7 etc.)
                if row1_count <= 1 and not is_p_in_0 and (hamming >= 4 or abs_idx > anchor['idx'] + 25):
                    confirmed_count += 1
                    lockout_timer = 8 # Safety: Ready for next floor by frame 9
                    cv2.imwrite(f"{OUT_DIR}/confirmed/F{confirmed_count:03}_Idx{abs_idx:05}_VAC.jpg", img_bgr)
                    print(f" >>> [VACUUM] F{confirmed_count} at {abs_idx:05} | H: {hamming}")
                    anchor = {"idx": abs_idx, "dna": this_dna, "hud": cur_hud.copy()}
                    continue

                # STRIKE (Catching fast transitions)
                if hud_diff > dynamic_thresh and hamming >= 4:
                    confirmed_count += 1
                    lockout_timer = 8 
                    cv2.imwrite(f"{OUT_DIR}/confirmed/F{confirmed_count:03}_Idx{abs_idx:05}_STRK.jpg", img_bgr)
                    print(f" >>> [STRIKE] F{confirmed_count} at {abs_idx:05} | H: {hamming}")
                    anchor = {"idx": abs_idx, "dna": this_dna, "hud": cur_hud.copy()}

        last_gray = img_gray.copy()
        if abs_idx % 500 == 0: 
            print(f" [PROGRESS] {abs_idx:05} | Floors: {confirmed_count} | Thresh: {dynamic_thresh:.1f}")

    executor.shutdown()
    print(f"\n[FINISH] Scanned {len(files)} frames. Total Floors: {confirmed_count}")

if __name__ == "__main__":
    run_v5_45_calibrated_auditor()