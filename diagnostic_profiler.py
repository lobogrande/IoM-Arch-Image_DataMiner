import cv2
import numpy as np
import os
import time

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]
PLAYER_ROI_Y = (120, 420)
ROW_0_MAX_Y = 300 
HEARTBEAT_INTERVAL = 250
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
    if is_text_heavy_slot: mask[15:34, :] = 0 
    return mask

def is_player_in_slot0(roi_gray, p_right, p_left):
    slot0_roi = roi_gray[:, 0:115]
    res_r = cv2.matchTemplate(slot0_roi, p_right, cv2.TM_CCOEFF_NORMED)
    res_l = cv2.matchTemplate(slot0_roi, p_left, cv2.TM_CCOEFF_NORMED)
    return max(cv2.minMaxLoc(res_r)[1], cv2.minMaxLoc(res_l)[1]) > 0.65

def is_xhair_present(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 120), (180, 255, 255))
    return cv2.countNonZero(mask) > 8

def get_slot_status_v456(roi_gray, full_img_bgr, rect, mask, templates, prev_bit="0", is_row1=False):
    # 1. Background Score
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    
    # 2. Xhair Check (Restored)
    if is_xhair_present(full_img_bgr[rect[1]:rect[3], rect[0]:rect[2]]): return "1"

    # 3. Ore Check
    ore_s = 0.0
    for t_list in templates['active']:
        s = cv2.matchTemplate(roi_gray, t_list[0], cv2.TM_CCORR_NORMED, mask=mask).max()
        if not is_row1 and s < 0.85:
            s = max(s, cv2.matchTemplate(roi_gray, t_list[1], cv2.TM_CCORR_NORMED, mask=mask).max())
        if s > ore_s: ore_s = s

    delta = ore_s - bg_s
    is_dirty = (delta > 0.065 if is_row1 else delta > 0.04) or (bg_s < 0.82)
    
    # Sticky Logic (Restored)
    if prev_bit == "1" and not is_row1:
        return "0" if (bg_s > 0.95 and delta < 0.02) else "1"
    return "1" if is_dirty else "0"

# --- MAIN ENGINE ---

def run_v5_20_high_fidelity_hybrid():
    buffer_root, out_dir = "capture_buffer_0", "production_audit_HYBRID"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)
    
    p_right, p_left = cv2.imread("templates/player_right.png", 0), cv2.imread("templates/player_left.png", 0)
    raw_ore_tpls = {'ore': {}, 'bg': []}
    
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): raw_ore_tpls['bg'].append(img)
        elif "_" in f and any(f.startswith(tier) for tier in KNOWN_TIERS):
            parts = f.replace(".png", "").split("_")
            tier, state = parts[0], parts[1]
            if tier not in raw_ore_tpls['ore']: raw_ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']:
                m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
                raw_ore_tpls['ore'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)
    
    # Root Initialization
    f1_bgr = cv2.imread(os.path.join(buffer_root, files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    anchor = {"num": 1, "idx": 0, "dna": "START", "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy()}
    cv2.imwrite(f"{out_dir}/confirmed/F001_Idx00000_ROOT.jpg", f1_bgr)
    
    dna_memory, confirmed_count, start_time = ["0"] * 24, 1, time.time()
    print(f"--- Running v5.20: High-Fidelity Hybrid Auditor ---")

    for i in range(1, len(files)):
        if confirmed_count >= 50: break
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Reactive Trigger Logic
        cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
        hud_diff = np.mean(cv2.absdiff(cur_hud, anchor['hud']))
        is_p_in_0 = is_player_in_slot0(img_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], :], p_right, p_left)

        if hud_diff > 3.8 or not is_p_in_0:
            # 1. Update Active Templates for THIS floor
            active_list = []
            allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= confirmed_count <= x]
            for tier in allowed:
                if tier in raw_ore_tpls['ore']:
                    for state in ['act', 'sha']: active_list.extend(raw_ore_tpls['ore'][tier][state])
            runtime_tpls = {'active': active_list, 'bg': raw_ore_tpls['bg']}

            # 2. Calculate DNA with Sticky Bits
            current_dna_bits, row1_clean = [], True
            for c in range(24):
                r, col = divmod(c, 6)
                x1, y1 = int(SLOT1_CENTER[0]+(col*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                bit = get_slot_status_v456(img_gray[y1:y1+48, x1:x1+48], img_bgr, (x1,y1,x1+48,y1+48), 
                                           text_mask if c in [2,3] else std_mask, runtime_tpls, 
                                           prev_bit=dna_memory[c], is_row1=(c<6))
                current_dna_bits.append(bit)
                if c < 6 and bit == "1": row1_clean = False
            
            this_dna = "".join(current_dna_bits)

            # CASE A: VACUUM
            if row1_clean and not is_p_in_0:
                if this_dna != anchor['dna'] or (i > anchor['idx'] + 25):
                    confirmed_count += 1
                    cv2.imwrite(f"{out_dir}/confirmed/F{confirmed_count:03}_Idx{i:05}_VAC.jpg", img_bgr)
                    print(f" >>> [VACUUM] F{confirmed_count} at {i:05}")
                    anchor = {"num": confirmed_count, "idx": i, "dna": this_dna, "hud": cur_hud.copy()}
                    dna_memory = current_dna_bits.copy()
                    continue

            # CASE B: STRIKE (with Interlock)
            if hud_diff > 3.8 and this_dna != anchor['dna']:
                res = cv2.matchTemplate(img_gray[150:500, 0:480], p_right, cv2.TM_CCOEFF_NORMED)
                _, _, _, ml = cv2.minMaxLoc(res)
                if (ml[1] + 150) <= ROW_0_MAX_Y:
                    confirmed_count += 1
                    cv2.imwrite(f"{out_dir}/confirmed/F{confirmed_count:03}_Idx{i:05}_STRK.jpg", img_bgr)
                    print(f" >>> [STRIKE] F{confirmed_count} at {i:05}")
                    anchor = {"num": confirmed_count, "idx": i, "dna": this_dna, "hud": cur_hud.copy()}
                    dna_memory = current_dna_bits.copy()

    print(f"\n[FINISH] Verified {confirmed_count} floors.")

if __name__ == "__main__":
    run_v5_20_high_fidelity_hybrid()