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
ROW_1_Y_ZONE = (150, 320)
ROW_0_MAX_Y = 300 

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- HELPERS FROM v4.56 ---
def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[15:34, :] = 0 
    return mask

def is_player_present_v456(roi_gray, p_right, p_left):
    roi = roi_gray.copy()
    roi[roi > 210] = 128
    best_score = 0
    for p_temp in [p_right, p_left]:
        s1 = cv2.minMaxLoc(cv2.matchTemplate(roi, p_temp, cv2.TM_CCOEFF_NORMED))[1]
        small_p = cv2.resize(p_temp, (0,0), fx=0.92, fy=0.92)
        s2 = cv2.minMaxLoc(cv2.matchTemplate(roi, small_p, cv2.TM_CCOEFF_NORMED))[1]
        best_score = max(best_score, s1, s2)
    return best_score > 0.65

def get_slot_status_v456(roi_gray, full_img_bgr, rect, mask, templates, prev_bit="0", is_row1=False):
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    ore_s = 0.0
    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for rots in types[state]:
                s = cv2.matchTemplate(roi_gray, rots[0], cv2.TM_CCORR_NORMED, mask=mask).max()
                if not is_row1 and s < 0.85:
                    s = max(s, cv2.matchTemplate(roi_gray, rots[1], cv2.TM_CCORR_NORMED, mask=mask).max())
                if s > ore_s: ore_s = s
    delta = ore_s - bg_s
    is_dirty = (delta > 0.065 if is_row1 else delta > 0.04) or (bg_s < 0.82)
    return "1" if is_dirty else "0"

# --- HELPERS FROM v35.0 ---
def is_slot_clean_v35(img_gray, slot_idx, templates, is_sliver=False):
    cx, cy = int(SLOT1_CENTER[0] + (slot_idx * STEP_X)), SLOT1_CENTER[1]
    roi = img_gray[cy-24:cy+24, cx-24:cx-12] if is_sliver else img_gray[cy-24:cy+24, cx-24:cx+24]
    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for rots in types[state]:
                t_roi = rots[0][:, :12] if is_sliver else rots[0]
                if cv2.minMaxLoc(cv2.matchTemplate(roi, t_roi, cv2.TM_CCOEFF_NORMED))[1] > 0.77:
                    return False
    return True

def get_ore_id_v21(img_gray, slot_idx, current_floor, templates):
    cx, cy = int(SLOT1_CENTER[0] + (slot_idx * STEP_X)), SLOT1_CENTER[1]
    roi = img_gray[cy-24:cy+24, cx-24:cx+24]
    audit_roi = roi[12:, :] if slot_idx in [2, 3] else roi
    allowed = [p for p, (m, x) in ORE_RESTRICTIONS.items() if m <= current_floor <= x]
    best = {'tier': 'empty', 'score': 0.0}
    for tier in allowed:
        for state in ['act', 'sha']:
            for rots in templates['ore'][tier][state]:
                t_audit = rots[0][12:, :] if slot_idx in [2, 3] else rots[0]
                score = cv2.minMaxLoc(cv2.matchTemplate(audit_roi, t_audit, cv2.TM_CCOEFF_NORMED))[1]
                if score > best['score']: best = {'tier': tier, 'score': score}
    return best['tier'] if best['score'] > 0.77 else "empty"

def run_v5_hybrid_auditor():
    buffer_root, out_dir = "capture_buffer_0", "production_audit_HYBRID"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)
    
    # LOAD TEMPLATES
    p_right, p_left = cv2.imread("templates/player_right.png", 0), cv2.imread("templates/player_left.png", 0)
    ore_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None: continue
        img = cv2.resize(img, (48, 48))
        if f.startswith("background"): ore_tpls['bg'].append(img)
        elif "_" in f:
            tier, state = f.split("_")[0], f.split("_")[1].replace(".png","")
            if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
            m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
            ore_tpls['ore'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)
    
    # INITIAL ANCHOR
    f1_gray = cv2.imread(os.path.join(buffer_root, files[0]), 0)
    anchor = {
        "num": 1, "idx": 0, "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]],
        "dna": None
    }
    dna_memory = ["0"] * 24
    
    print(f"--- Running v5.0: Hybrid Logic Auditor ---")

    for i in range(1, len(files)):
        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # --- LOGIC 1: v4.56 VACUUM GATE (For mid-row starts) ---
        if not is_player_present_v456(img_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], :], p_right, p_left):
            current_bits, row1_clean = [], True
            for c in range(6):
                x1, y1 = int(SLOT1_CENTER[0]+(c*STEP_X))-24, int(SLOT1_CENTER[1])-24
                bit = get_slot_status_v456(img_gray[y1:y1+48, x1:x1+48], img_bgr, (x1,y1,x1+48,y1+48), 
                                           text_mask if c in [2,3] else std_mask, ore_tpls, is_row1=True)
                current_bits.append(bit)
                if bit == "1": row1_clean = False; break
            
            if row1_clean:
                # Profiling for DNA Stability
                for s_idx in range(6, 24):
                    r, c = divmod(s_idx, 6)
                    x1, y1 = int(SLOT1_CENTER[0]+(c*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                    bit = get_slot_status_v456(img_gray[y1:y1+48, x1:x1+48], img_bgr, (x1,y1,x1+48,y1+48), 
                                               std_mask, ore_tpls, prev_bit=dna_memory[s_idx], is_row1=False)
                    current_bits.append(bit)
                    dna_memory[s_idx] = bit
                
                this_dna = "".join(current_bits)
                if this_dna != anchor['dna'] or (i > anchor['idx'] + 25):
                    # COMMIT & SYNC
                    f_num = anchor['num']
                    cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{i:05}.jpg", img_bgr)
                    print(f" [VACUUM] Floor {f_num} detected via DNA at Index {i}")
                    anchor = {"num": f_num + 1, "idx": i, "hud": img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]], "dna": this_dna}
                continue

        # --- LOGIC 2: v35.0 PATH AUDIT FALLBACK (Standard floors) ---
        res = cv2.matchTemplate(img_gray[150:500, 0:480], p_right, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        actual_y = max_loc[1] + 150
        
        if actual_y <= ROW_0_MAX_Y:
            cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            if np.mean(cv2.absdiff(cur_hud, anchor['hud'])) > 3.5:
                n_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
                if n_slot is not None:
                    path_clean = True
                    if n_slot > 0 and not is_slot_clean_v35(img_gray, n_slot - 1, ore_tpls, True): path_clean = False
                    if path_clean and n_slot >= 2:
                        for s_idx in range(n_slot - 1):
                            if not is_slot_clean_v35(img_gray, s_idx, ore_tpls, False): path_clean = False; break
                    
                    if path_clean:
                        f_num = anchor['num']
                        cv2.imwrite(f"{out_dir}/confirmed/F{f_num:03}_Idx{i:05}.jpg", img_bgr)
                        print(f" [STRIKE] Floor {f_num} detected via Path at Index {i}")
                        anchor = {"num": f_num + 1, "idx": i, "hud": cur_hud.copy(), "dna": None}
                        dna_memory = ["0"] * 24

if __name__ == "__main__":
    run_v5_hybrid_auditor()