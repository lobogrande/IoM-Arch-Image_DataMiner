import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# --- FORENSIC CONTROL ---
START_INDEX = 0      
END_INDEX = 2500     
START_FLOOR = 1      
OUT_DIR = "verified_audit_v553"

# --- CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
PLAYER_ROI_Y = (120, 420)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

# --- GROUND TRUTH ---
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

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:24, :] = 0 
    return mask

def get_slot_status_worker(args):
    roi, mask, templates, is_row1 = args
    bg_s = max([cv2.matchTemplate(roi, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    ore_s = max([cv2.matchTemplate(roi, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['ore_all']])
    delta = ore_s - bg_s
    thresh = 0.08 if is_row1 else 0.05
    return "1" if (delta > thresh or bg_s < 0.83) else "0"

# --- MAIN ENGINE ---

def run_v5_53_verified_auditor():
    buffer_root = "capture_buffer_0"
    os.makedirs(f"{OUT_DIR}/confirmed", exist_ok=True)
    
    # 1. LOAD TEMPLATES
    raw_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        if f.startswith('.') or not f.lower().endswith('.png'): continue
        img = cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48))
        if f.startswith("background"): raw_tpls['bg'].append(img)
        elif "_" in f:
            tier = f.split("_")[0]
            if tier not in raw_tpls['ore']: raw_tpls['ore'][tier] = []
            raw_tpls['ore'][tier].append(img)

    all_files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    files = all_files[START_INDEX:END_INDEX] if END_INDEX else all_files[START_INDEX:]
    std_mask, txt_mask = get_combined_mask(False), get_combined_mask(True)
    
    # 2. INITIALIZE ANCHOR
    first_img_bgr = cv2.imread(os.path.join(buffer_root, files[0]))
    last_gray = cv2.cvtColor(first_img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Pre-calc DNA for Frame 0
    allowed_start = [t for t, (low, high) in ORE_RESTRICTIONS.items() if low <= START_FLOOR <= high]
    init_ores = []
    for t in allowed_start: init_ores.extend(raw_tpls['ore'].get(t, []))
    init_tpls = {'ore_all': init_ores, 'bg': raw_tpls['bg']}
    init_dna = []
    for c in range(24):
        r, col = divmod(c, 6)
        x, y = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
        init_dna.append(get_slot_status_worker((last_gray[y:y+48, x:x+48], txt_mask if c in [2,3] else std_mask, init_tpls, (c<6))))
    
    anchor = {"idx": START_INDEX, "dna": "".join(init_dna), "hud": last_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy()}
    
    if START_INDEX == 0:
        cv2.imwrite(f"{OUT_DIR}/confirmed/F{START_FLOOR:03}_Idx{START_INDEX:05}_ROOT.jpg", first_img_bgr)

    dna_memory = list(anchor['dna'])
    confirmed_count = START_FLOOR
    lockout, persistence = 0, 0
    last_dna = "".join(dna_memory)
    history = deque(maxlen=8) # Buffer for backtracking
    
    executor = ThreadPoolExecutor(max_workers=24)
    start_time = time.time()

    print(f"--- Running v5.53: VERIFIED Auditor ---")

    for i in range(1, len(files)):
        abs_idx = START_INDEX + i
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(cur_hud, anchor['hud']))

        if lockout > 0: lockout -= 1

        # Gated Permission: Only calculate if board is moving or HUD shifted
        pixel_move = np.mean(cv2.absdiff(img_gray[200:500, :], last_gray[200:500, :]))
        if (hud_diff > 2.0 or pixel_move > 0.8) and lockout == 0:
            allowed_tiers = [t for t, (low, high) in ORE_RESTRICTIONS.items() if low <= confirmed_count <= high]
            ore_all = []
            for t in allowed_tiers:
                if t in raw_tpls['ore']: ore_all.extend(raw_tpls['ore'][t])
            runtime_tpls = {'ore_all': ore_all, 'bg': raw_tpls['bg']}

            tasks, task_indices = [], []
            for c in range(24):
                r, col = divmod(c, 6)
                x, y = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                if np.mean(cv2.absdiff(img_gray[y:y+48, x:x+48], last_gray[y:y+48, x:x+48])) < 0.2: continue
                tasks.append((img_gray[y:y+48, x:x+48], txt_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
                task_indices.append(c)

            if tasks:
                results = list(executor.map(get_slot_status_worker, tasks))
                for idx, res in zip(task_indices, results): dna_memory[idx] = res
            
            curr_dna = "".join(dna_memory)
            history.append((abs_idx, img_bgr.copy(), curr_dna))
            
            # PERSISTENCE COUNTER
            if curr_dna == last_dna: persistence += 1
            else: persistence = 0
            last_dna = curr_dna

            # --- THE VERIFIED TRIGGER ---
            # Condition 1: Stable for 3 frames (persistence >= 2)
            # Condition 2: HUD shift OR Hamming jump
            if persistence >= 2 and lockout == 0:
                hamming = sum(c1 != c2 for c1, c2 in zip(anchor['dna'], curr_dna))
                row1_count = curr_dna[:6].count("1")

                # STRIKE or VACUUM (Both now require Hamming verification)
                if (hud_diff > 3.0 and hamming >= 4) or (row1_count <= 1 and hamming >= 5):
                    # BACKTRACK: Find the arrival frame
                    alpha_idx, alpha_img = abs_idx, img_bgr
                    for h_idx, h_img, h_dna in reversed(list(history)):
                        if h_dna == curr_dna:
                            alpha_idx, alpha_img = h_idx, h_img
                        else: break

                    confirmed_count += 1
                    lockout = 8 
                    cv2.imwrite(f"{OUT_DIR}/confirmed/F{confirmed_count:03}_Idx{alpha_idx:05}_VER.jpg", alpha_img)
                    print(f" >>> [VER] F{confirmed_count} at {alpha_idx:05} | H: {hamming} | HUD: {hud_diff:.1f}")
                    anchor = {"idx": alpha_idx, "dna": curr_dna, "hud": cur_hud.copy()}

        last_gray = img_gray.copy()
        if abs_idx % 500 == 0: 
            print(f" [PROGRESS] {abs_idx:05} | Floors: {confirmed_count} | {time.time()-start_time:.1f}s")

    executor.shutdown()
    print(f"\n[FINISH] Total Floors Found: {confirmed_count}")

if __name__ == "__main__":
    run_v5_53_verified_auditor()