import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v52_Forensic_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SURGICAL CONSTANTS
MATCH_THRESHOLD = 0.85  # Slightly lowered for troubleshooting
OFFSET_X = 24           
D_GATE_LIVE = 6.5       # Ultra-sensitive for early Dirt1
D_GATE_SHADOW = 3.5     

def get_slot_state(roi, bg_template):
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    if diff > D_GATE_LIVE: return 2
    if diff > D_GATE_SHADOW: return 1
    return 0

def run_v52_forensic_audit():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    last_logged_dna = None

    print(f"--- Running v5.2 Forensic Master Auditor ---")
    print(f"Total Frames to scan: {len(buffer_files)}")

    for i in range(len(buffer_files) - 1):
        if i % 100 == 0:
            print(f" [Heartbeat] Frame {i} | Floors Found: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. TEMPLATE SEEK
        search_roi = img_n1_gray[200:350, 0:450]
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. ANCHOR LOGIC
            player_center_x = max_loc[0] + 20 
            target_col = round((player_center_x + OFFSET_X - 74) / 59.1)
            
            if target_col in range(6):
                cy = 261
                tx_start = int(74+(target_col*59.1))-5
                target_roi = img_n1_gray[cy-5:cy+5, tx_start:tx_start+10]
                target_state = get_slot_state(target_roi, bg_t[0])
                
                # REJECTION REASON 1: Target Slot is Empty
                if target_state != 2:
                    continue

                # REJECTION REASON 2: Left Gutter is Dirty (Shadows/Ores exist)
                left_is_clean = True
                for l_col in range(target_col):
                    lcx = int(74 + (l_col * 59.1))
                    if get_slot_state(img_n1_gray[cy-5:cy+5, lcx-5:lcx+5], bg_t[0]) > 0:
                        left_is_clean = False
                        break
                
                if not left_is_clean:
                    continue
                
                # REJECTION REASON 3: DNA Persistence (Same Floor)
                # We check Row 1 only for the skip-logic to stay fast
                current_row1_dna = [get_slot_state(img_n1_gray[cy-5:cy+5, int(74+(c*59.1))-5:int(74+(c*59.1))+5], bg_t[0]) for c in range(6)]
                if current_row1_dna == last_logged_dna:
                    continue

                # --- SUCCESSFUL TRIGGER ---
                floor_num = len(floor_library) + 1
                last_logged_dna = current_row1_dna
                
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                cv2.putText(bgr_n, f"F{i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_n1, f"F{i+1} (START FLOOR {floor_num})", (30, 50), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                floor_library.append({"floor": floor_num, "idx": i+1, "score": round(max_val, 3)})
                
                print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Col: {target_col} | Score: {max_val:.3f}")

    print(f"\n[FINISH] Auditor mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v52_forensic_audit()