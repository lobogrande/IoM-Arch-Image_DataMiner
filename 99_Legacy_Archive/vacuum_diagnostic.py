import cv2
import numpy as np
import os
import time

# --- PRODUCTION CONSTANTS ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
ROW_1_Y_ZONE = (150, 320)
PLAYER_ROI_Y = (120, 420)
HEARTBEAT_INTERVAL = 250 # Progress tracking frequency

# --- SCAN CONFIGURATION ---
SCAN_START = 0
SCAN_LIMIT = 20000

def get_combined_mask(is_text_heavy_slot=False):
    mask = np.zeros((AI_DIM, AI_DIM), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    if is_text_heavy_slot: mask[15:34, :] = 0 
    return mask

def load_all_templates():
    templates = {'block': {}, 'bg': []}
    t_path = "templates"
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img = cv2.resize(img, (AI_DIM, AI_DIM))
        if f.startswith("background"): templates['bg'].append(img)
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1].replace(".png", "")
            if tier not in templates['block']: templates['block'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']:
                m5 = cv2.getRotationMatrix2D((24, 24), 5, 1.0)
                templates['block'][tier][state].append([img, cv2.warpAffine(img, m5, (48, 48))])
    return templates

def is_player_present(full_gray, p_right, p_left):
    # Noise Filter: Mask out pure white damage numbers
    roi = full_gray[PLAYER_ROI_Y[0]:PLAYER_ROI_Y[1], 0:350].copy()
    roi[roi > 210] = 128
    
    best_score = 0
    for p_temp in [p_right, p_left]:
        # Check standard and 92% scale for recoil compression
        s1 = cv2.minMaxLoc(cv2.matchTemplate(roi, p_temp, cv2.TM_CCOEFF_NORMED))[1]
        small_p = cv2.resize(p_temp, (0,0), fx=0.92, fy=0.92)
        s2 = cv2.minMaxLoc(cv2.matchTemplate(roi, small_p, cv2.TM_CCOEFF_NORMED))[1]
        best_score = max(best_score, s1, s2)
    return best_score > 0.65

def is_crosshair_present(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 120), (180, 255, 255))
    return cv2.countNonZero(mask) > 8

def get_slot_status(roi_gray, full_img_bgr, rect, mask, templates, prev_bit="0", is_row1=False):
    bg_s = max([cv2.matchTemplate(roi_gray, bg, cv2.TM_CCORR_NORMED, mask=mask).max() for bg in templates['bg']])
    
    if is_row1:
        block_s = 0.0
        for tier in templates['block']:
            for state in ['act', 'sha']:
                for rots in templates['block'][tier][state]:
                    s = cv2.matchTemplate(roi_gray, rots[0], cv2.TM_CCORR_NORMED, mask=mask).max()
                    if s > block_s: block_s = s
        return "1" if (ore_s - bg_s > 0.065 or bg_s < 0.82) else "0"

    x1, y1, x2, y2 = rect
    if is_crosshair_present(full_img_bgr[y1:y2, x1:x2]): return "1"

    block_s = 0.0
    for tier in templates['block']:
        for state in ['act', 'sha']:
            for rots in templates['block'][tier][state]:
                s = max(cv2.matchTemplate(roi_gray, rots[0], cv2.TM_CCORR_NORMED, mask=mask).max(),
                        cv2.matchTemplate(roi_gray, rots[1], cv2.TM_CCORR_NORMED, mask=mask).max())
                if s > block_s: block_s = s

    delta = block_s - bg_s
    if prev_bit == "0":
        return "1" if (delta > 0.04) else "0"
    else:
        return "0" if (bg_s > 0.95 and delta < 0.02) else "1"

def run_v4_56_master_auditor():
    buffer_root, out_dir = "capture_buffer_0", "production_audit_Run0"
    os.makedirs(out_dir, exist_ok=True)
    templates = load_all_templates()
    p_right, p_left = cv2.imread("templates/player_right.png", 0), cv2.imread("templates/player_left.png", 0)
    std_mask, text_mask = get_combined_mask(False), get_combined_mask(True)
    files = sorted([f for f in os.listdir(buffer_root) if f.lower().endswith(('.png', '.jpg'))])
    
    floor_count, last_dna, last_idx = 0, None, -1
    dna_memory = ["0"] * 24
    start_time = time.time()

    print(f"--- Running v4.56: Master Auditor [Lava Biome / Run_0] ---")

    for i in range(SCAN_START, min(SCAN_LIMIT, len(files))):
        # Progress Heartbeat
        if i % HEARTBEAT_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f" [PROGRESS] Scanned: {i:05} | Elapsed: {elapsed:.1f}s | Floors: {floor_count}")

        img_gray = cv2.imread(os.path.join(buffer_root, files[i]), 0)
        
        # 1. Player Gate (Noise-Filtered)
        if is_player_present(img_gray, p_right, p_left):
            dna_memory = ["0"] * 24
            continue 

        # 2. Row 1 Vacuum Gate
        img_bgr_full = cv2.imread(os.path.join(buffer_root, files[i]))
        current_bits, row1_clean = [], True
        for c in range(6):
            x1, y1 = int(SLOT1_CENTER[0]+(c*STEP_X))-24, int(SLOT1_CENTER[1])-24
            bit = get_slot_status(img_gray[y1:y1+48, x1:x1+48], img_bgr_full, (x1,y1,x1+48,y1+48), 
                                  text_mask if c in [2,3] else std_mask, templates, is_row1=True)
            current_bits.append(bit)
            if bit == "1": row1_clean = False; break
        
        if not row1_clean: continue

        # 3. DNA Profiling
        for s_idx in range(6, 24):
            r, c = divmod(s_idx, 6)
            x1, y1 = int(SLOT1_CENTER[0]+(c*STEP_X))-24, int(SLOT1_CENTER[1])+(r*int(STEP_Y))-24
            bit = get_slot_status(img_gray[y1:y1+48, x1:x1+48], img_bgr_full, (x1,y1,x1+48,y1+48), 
                                  std_mask, templates, prev_bit=dna_memory[s_idx], is_row1=False)
            current_bits.append(bit)
            dna_memory[s_idx] = bit 

        this_dna = "".join(current_bits)
        if this_dna != last_dna or (i > last_idx + 25):
            floor_count += 1
            last_dna = this_dna
            dna_str = "|".join([this_dna[k:k+6] for k in range(0,24,6)])
            print(f" >>> [HIT] Idx {i:05} | DNA: {dna_str}")
        
        last_idx = i
        cv2.imwrite(os.path.join(out_dir, f"Idx{i:05}.jpg"), img_bgr_full)

    total_time = time.time() - start_time
    print(f"\n[FINISH] Scanned {len(files)} frames in {total_time:.1f}s. Unique floors found: {floor_count}")

if __name__ == "__main__":
    run_v4_56_master_auditor()