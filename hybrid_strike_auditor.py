import cv2
import numpy as np
import os
import time

# --- DEBUG CONTROL ---
TARGET_FLOOR_LIMIT = 110 
SCAN_LIMIT = 20000

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
HEADER_ROI = (54, 74, 103, 138)
VALID_X_ANCHORS = [11, 70, 129, 188, 247, 306]
PLAYER_ROI_Y = (120, 420)
ROW_1_Y_ZONE = (150, 320)
ROW_0_MAX_Y = 300 
HEARTBEAT_INTERVAL = 250

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
    """Checks only the leftmost slot area for player presence."""
    # Slot 0 area is roughly x:0 to x:120
    slot0_roi = roi_gray[:, 0:120]
    res_r = cv2.matchTemplate(slot0_roi, p_right, cv2.TM_CCOEFF_NORMED)
    res_l = cv2.matchTemplate(slot0_roi, p_left, cv2.TM_CCOEFF_NORMED)
    return max(cv2.minMaxLoc(res_r)[1], cv2.minMaxLoc(res_l)[1]) > 0.65

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
    
    # Sticky logic for DNA stability
    if prev_bit == "1" and not is_row1:
        return "0" if (bg_s > 0.95 and delta < 0.02) else "1"
    return "1" if is_dirty else "0"

def is_slot_clean_v35(img_gray, slot_idx, templates, is_sliver=False):
    cx, cy = int(SLOT1_CENTER[0] + (slot_idx * STEP_X)), int(SLOT1_CENTER[1])
    roi = img_gray[cy-24:cy+24, cx-24:cx-12] if is_sliver else img_gray[cy-24:cy+24, cx-24:cx+24]
    for tier, types in templates['ore'].items():
        for state in ['act', 'sha']:
            for rots in types[state]:
                t_roi = rots[0][:, :12] if is_sliver else rots[0]
                if cv2.minMaxLoc(cv2.matchTemplate(roi, t_roi, cv2.TM_CCOEFF_NORMED))[1] > 0.77:
                    return False
    return True

def get_ore_id_v21(img_gray, slot_idx, current_floor, templates):
    cx, cy = int(SLOT1_CENTER[0] + (slot_idx * STEP_X)), int(SLOT1_CENTER[1])
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

# --- MAIN ENGINE ---

def run_v5_07_local_gate_hybrid():
    buffer_root, out_dir = "capture_buffer_0", "production_audit_HYBRID"
    os.makedirs(f"{out_dir}/confirmed", exist_ok=True)
    
    p_right, p_left = cv2.imread("templates/player_right.png", 0), cv2.imread("templates/player_left.png", 0)
    ore_tpls = {'ore': {}, 'bg': []}
    for f in os.listdir("templates"):
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is None or f.startswith("background") or "_" not in f: 
            if img is not None and f.startswith("background"): ore_tpls['bg'].append(cv2.resize(img, (48, 48)))
            continue
        parts = f.replace(".png", "").split("_")
        tier, state = parts[0], parts[1]
        if tier not in ore_tpls['ore']: ore_tpls['ore'][tier] = {'act': [], 'sha': []}
        if state in ['act', 'sha']:
            img = cv2.resize(img, (48, 48)); m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
            ore_tpls['ore'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])

    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)
    
    # ROOT START
    f1_bgr = cv2.imread(os.path.join(buffer_root, files[0]))
    f1_gray = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    initial_ore = get_ore_id_v21(f1_gray, 0, 1, ore_tpls)
    anchor = {"num": 1, "idx": 0, "hud": f1_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]], "dna": None, "ore": initial_ore}
    cv2.imwrite(f"{out_dir}/confirmed/F001_Idx00000_ROOT.jpg", f1_bgr)
    
    dna_memory, confirmed_count, start_time = ["0"] * 24, 1, time.time()

    print(f"--- Running v5.07: Local-Gate Hybrid Auditor ---")

    for i in range(1, len(files)):
        if confirmed_count >= TARGET_FLOOR_LIMIT: break
        if i % HEARTBEAT_INTERVAL == 0:
            print(f" [PROGRESS] {i:05} | Total Floors: {confirmed_count}")

        img_bgr = cv2.imread(os.path.join(buffer_root, files[i]))
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. PRIMARY: [VACUUM] CHECK
        # Only gate if player is in Slot 0. If he's moving right, Vacuum is open!
        if not is_player_in_slot0(img_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], :], p_right, p_left):
            current_bits, row1_clean = [], True
            for c in range(6):
                x, y = int(SLOT1_CENTER[0]+(c*STEP_X))-24, int(SLOT1_CENTER[1])-24
                bit = get_slot_status_v456(img_gray[y:y+48, x:x+48], img_bgr, (x,y,x+48,y+48), 
                                           text_mask if c in [2,3] else std_mask, ore_tpls, is_row1=True)
                current_bits.append(bit)
                if bit == "1": row1_clean = False; break
            
            if row1_clean:
                for s_idx in range(6, 24):
                    r, c = divmod(s_idx, 6)
                    x, y = int(SLOT1_CENTER[0]+(c*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
                    bit = get_slot_status_v456(img_gray[y:y+48, x:x+48], img_bgr, (x,y,x+48,y+48), 
                                               std_mask, ore_tpls, prev_bit=dna_memory[s_idx], is_row1=False)
                    current_bits.append(bit); dna_memory[s_idx] = bit
                
                this_dna = "".join(current_bits)
                if this_dna != anchor['dna'] or (i > anchor['idx'] + 20):
                    confirmed_count += 1
                    cv2.imwrite(f"{out_dir}/confirmed/F{confirmed_count:03}_Idx{i:05}_VAC.jpg", img_bgr)
                    print(f" >>> [VACUUM] F{confirmed_count:03} at {i:05} | DNA: {'|'.join([this_dna[k:k+6] for k in range(0,24,6)])}")
                    anchor = {"num": confirmed_count, "idx": i, "hud": img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]].copy(), "dna": this_dna, "ore": "multi"}
                    continue

        # 2. SECONDARY: [STRIKE] CHECK
        res = cv2.matchTemplate(img_gray[150:500, 0:480], p_right, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        if (max_loc[1] + 150) <= ROW_0_MAX_Y:
            cur_hud = img_gray[HEADER_ROI[0]:HEADER_ROI[1], HEADER_ROI[2]:HEADER_ROI[3]]
            if np.mean(cv2.absdiff(cur_hud, anchor['hud'])) > 3.8:
                n_slot = next((idx for idx, a in enumerate(VALID_X_ANCHORS) if abs(max_loc[0] - a) <= 15), None)
                if n_slot is not None:
                    if is_slot_clean_v35(img_gray, n_slot-1, ore_tpls, True) if n_slot > 0 else True:
                        confirmed_count += 1
                        ore_id = get_ore_id_v21(img_gray, n_slot, confirmed_count, ore_tpls)
                        cv2.imwrite(f"{out_dir}/confirmed/F{confirmed_count:03}_Idx{i:05}_STRK.jpg", img_bgr)
                        print(f" >>> [STRIKE] F{confirmed_count:03} at {i:05} ({ore_id})")
                        anchor = {"num": confirmed_count, "idx": i, "hud": cur_hud.copy(), "dna": None, "ore": ore_id}
                        dna_memory = ["0"] * 24

if __name__ == "__main__":
    run_v5_07_local_gate_hybrid()