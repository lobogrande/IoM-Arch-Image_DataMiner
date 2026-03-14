import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v83_Consensus"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.92

def is_banner_present(img_gray):
    """Detects if a horizontal scrolling banner is overlapping the HUD."""
    # ROI for the banner path
    banner_roi = img_gray[60:110, 0:480]
    # Banners are high-contrast; we look for a horizontal edge density spike
    edges = cv2.Sobel(banner_roi, cv2.CV_64F, 1, 0, ksize=3)
    return np.mean(np.abs(edges)) > 15 # Threshold for 'busy' scrolling text

def get_slot_dna(img_gray, col):
    """Returns the DNA signal for a specific slot in Row 1."""
    cx, cy = int(74 + (col * 59.1)), 261
    roi = img_gray[cy-8:cy+8, cx-8:cx+8]
    return np.mean(roi)

def run_v83_consensus_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_x = -999
    last_logged_slot_dna = -1

    print(f"--- Running v8.3 Consensus Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. SEEK PLAYER
        search_roi = img_n1_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            
            # 2. CHECK FOR ANCHOR SNAP
            anchor_col = None
            for idx, a in enumerate(VALID_ANCHORS):
                if abs(current_x - a) <= 3:
                    anchor_col = idx
                    break
            
            if anchor_col is not None and current_x != last_x:
                # 3. CONSENSUS CHECK: DNA + Banner Logic
                current_slot_dna = get_slot_dna(img_n1_gray, anchor_col)
                banner_blocking = is_banner_present(img_n1_gray)
                
                # If there's no banner, we can trust the DNA more
                dna_threshold = 5.0 if not banner_blocking else 8.0
                
                if abs(current_slot_dna - last_logged_slot_dna) > dna_threshold:
                    floor_num = len(floor_library) + 1
                    last_logged_slot_dna = current_slot_dna
                    
                    bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                    bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                    
                    cv2.putText(bgr_n, f"F{i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                    cv2.putText(bgr_n1, f"F{i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                    
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                    floor_library.append({"floor": floor_num, "idx": i+1})
                    
                    print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Banner: {banner_blocking}")

            last_x = current_x

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v83_consensus_audit()