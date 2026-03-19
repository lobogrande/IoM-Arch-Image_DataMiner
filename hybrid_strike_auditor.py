import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor

# --- FORENSIC CONTROL ---
START_INDEX = 0      
END_INDEX = 2500     
START_FLOOR = 1      
OUT_DIR = "final_audit_v548"

# --- CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
PLAYER_ROI_Y = (120, 420)
KNOWN_TIERS = ['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- HELPERS ---

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[0:28, :] = 0 # Calibrated Text Mask
    return mask

def get_slot_status_worker(args):
    roi_gray, mask, templates, is_row1 = args
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    ore_s = max([cv2.matchTemplate(roi_gray, ore, cv2.TM_CCORR_NORMED, mask=mask).max() for ore in templates['active']])
    delta = ore_s - bg_s
    # Reverted to safer 0.05 / 0.07 thresholds
    thresh = 0.07 if is_row1 else 0.05
    return "1" if (delta > thresh or bg_s < 0.81) else "0"

# --- MAIN ENGINE ---

def run_v5_48_final_auditor():
    buffer_root = "capture_buffer_0"
    os.makedirs(f"{OUT_DIR}/confirmed", exist_ok=True)
    p_right = cv2.imread("templates/player_right.png", 0)
    
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
    
    first_img_bgr = cv2.imread(os.path.join(buffer_root, files[0]))
    last_gray = cv2.cvtColor(first_img_bgr, cv2.COLOR_BGR2GRAY)
    anchor = {"idx": START_INDEX, "dna": "0"*24, "hud": last_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy()}
    
    if START_INDEX == 0:
        cv2.imwrite(f"{OUT_DIR}/confirmed/F001_Idx00000_ROOT.jpg", first_img_bgr)

    dna_memory, confirmed_count = ["0"] * 24, START_FLOOR
    lockout, persistence = 0, 0
    last_dna = "0"*24
    potential_start_idx = START_INDEX

    executor = ThreadPoolExecutor(max_workers=24)
    print(f"--- Running v5.48: FINAL Calibrated Auditor ---")

    for i in range(1, len(files)):
        abs_idx = START_INDEX + i
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(cur_hud, anchor['hud']))

        # Player Col Detection
        res_r = cv2.matchTemplate(img_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], :], p_right, cv2.TM_CCOEFF_NORMED)
        player_col = int((cv2.minMaxLoc(res_r)[3][0] - 74 + 24) / 59.1)

        if lockout > 0: lockout -= 1

        # LIGHT GATE: Always allow DNA math if not in lockout
        pixel_move = np.mean(cv2.absdiff(img_gray[200:500, :], last_gray[200:500, :]))
        if (hud_diff > 2.0 or pixel_move > 0.5) and lockout == 0:
            allowed = [t for t, (low, high) in ORE_RESTRICTIONS.items() if low <= confirmed_count <= high]
            active_list = []
            for tier in allowed:
                if tier in raw_tpls['ore']: active_list.extend(raw_tpls['ore'][tier])
            runtime_tpls = {'active': active_list, 'bg': raw_tpls['bg']}

            tasks, task_map = [], []
            for c in range(24):
                r, col = divmod(c, 6)
                x, y = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                # Perform delta check to skip static slots
                roi_now, roi_prev = img_gray[y:y+48, x:x+48], last_gray[y:y+48, x:x+48]
                if np.mean(cv2.absdiff(roi_now, roi_prev)) < 0.2: continue
                tasks.append((roi_now, txt_mask if c in [2,3] else std_mask, runtime_tpls, (c<6)))
                task_map.append(c)

            if tasks:
                results = list(executor.map(get_slot_status_worker, tasks))
                for idx, res in zip(task_map, results): dna_memory[idx] = res
            
            curr_dna = "".join(dna_memory)
            if curr_dna == last_dna: persistence += 1
            else:
                persistence = 0
                potential_start_idx = abs_idx
            last_dna = curr_dna

            bits_changed = [k for k in range(24) if anchor['dna'][k] != curr_dna[k]]
            hamming = len(bits_changed)
            
            # Smart Lane Veto: Only veto if Hamming is low
            is_veto = False
            if hamming < 3:
                is_veto = all((k % 6) == player_col for k in bits_changed)

            # --- CONFIRMATION TRIGGERS ---
            if lockout == 0 and not is_veto:
                # Trigger A: HUD Pulse (Stage change)
                if hud_diff > 4.5 and hamming >= 2 and persistence >= 1:
                    confirmed_count += 1
                    lockout = 9 # Solid 10-frame window
                    cv2.imwrite(f"{OUT_DIR}/confirmed/F{confirmed_count:03}_Idx{potential_start_idx:05}_HUD.jpg", img_bgr)
                    print(f" >>> [HUD] F{confirmed_count} at {potential_start_idx:05} | H: {hamming}")
                    anchor = {"idx": abs_idx, "dna": curr_dna, "hud": cur_hud.copy()}
                    continue

                # Trigger B: Massive Hamming (Quake/Velocity)
                if hamming >= 8 and persistence >= 2:
                    confirmed_count += 1
                    lockout = 9
                    cv2.imwrite(f"{OUT_DIR}/confirmed/F{confirmed_count:03}_Idx{potential_start_idx:05}_VEL.jpg", img_bgr)
                    print(f" >>> [VEL] F{confirmed_count} at {potential_start_idx:05} | H: {hamming}")
                    anchor = {"idx": abs_idx, "dna": curr_dna, "hud": cur_hud.copy()}

        last_gray = img_gray.copy()
        if abs_idx % 250 == 0: print(f" [PROGRESS] {abs_idx:05} | Floors: {confirmed_count}")

    executor.shutdown()
    print(f"\n[FINISH] Scanned {len(files)} frames. Total Floors: {confirmed_count}")

if __name__ == "__main__":
    run_v5_48_final_auditor()