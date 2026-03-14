import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v66_Sweep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.88  
OFFSET_X = 24           
D_GATE_LIVE = 6.0 

def get_best_dna_diff(img_gray, target_x, target_y, bg_template):
    """Scans a small 10px area to find the strongest ore signal."""
    max_diff = 0
    # Search +/- 5 pixels to find the true center of the ore
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            roi = img_gray[target_y+dy-5:target_y+dy+5, target_x+dx-5:target_x+dx+5]
            diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
            if diff > max_diff: max_diff = diff
    return max_diff

def run_v66_sweep_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # FORCED START - MUST BE HERE
    floor_library = [{"floor": 1, "idx": 0}]
    bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), bgr_start)
    
    last_logged_idx = 0

    print(f"--- Running v6.6 Coordinate-Sweep Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 100 == 0:
            print(f" [Scanning] Frame {i} | Found: {len(floor_library)}", end='\r')

        img_now = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_now is None: continue

        # 1. FIND PLAYER
        search_roi = img_now[230:310, 0:450]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. CALCULATE ANCHOR
            # max_loc[1] is relative to ROI (230). 
            # We add 30 to get to the center of the 60px template.
            p_center_x = max_loc[0] + 20
            p_center_y = max_loc[1] + 230 + 30 
            
            target_ore_x = p_center_x + OFFSET_X
            target_ore_y = 261 # Core grid Y
            
            # 3. JITTER SEARCH THE ORE
            dna_signal = get_best_dna_diff(img_now, int(target_ore_x), target_ore_y, bg_t[0])
            
            if dna_signal > D_GATE_LIVE:
                # 4. LEFT GUTTER CHECK (DNA based)
                col = round((target_ore_x - 74) / 59.1)
                left_dirty = False
                for l_col in range(col):
                    lx = int(74 + (l_col * 59.1))
                    if get_best_dna_diff(img_now, lx, 261, bg_t[0]) > D_GATE_LIVE:
                        left_dirty = True
                        break
                
                if not left_dirty and (i - last_logged_idx) > 5:
                    floor_num = len(floor_library) + 1
                    last_logged_idx = i
                    
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                    
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Signal: {dna_signal:.2f}")

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v66_sweep_audit()