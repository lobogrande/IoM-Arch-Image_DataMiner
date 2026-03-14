import cv2
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v51_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SURGICAL CONSTANTS
MATCH_THRESHOLD = 0.90  
OFFSET_X = 24           
D_GATE_LIVE = 7.0       # Sensitized for early-game Dirt1
D_GATE_SHADOW = 3.5     

def get_slot_state(roi, bg_template):
    """0=Gravel, 1=Shadow, 2=Live Ore"""
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    if diff > D_GATE_LIVE: return 2
    if diff > D_GATE_SHADOW: return 1
    return 0

def run_v51_master_audit():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    last_logged_dna = None

    print(f"--- Running v5.1 Master Auditor (Run_{TARGET_RUN}) ---")

    for i in range(len(buffer_files) - 1):
        # 1. SPECIAL CASE: START
        if i == 0:
            floor_library.append({"floor": 1, "idx": 0})
            bgr_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
            cv2.putText(bgr_start, "DATASET START - FLOOR 1", (30, 50), 0, 0.7, (0,255,0), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "START_Floor1.jpg"), bgr_start)
            continue

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 2. TEMPLATE SEEK: Player facing right in Row 1
        search_roi = img_n1_gray[230:310, 0:400]
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # Player center (max_loc[0] + half width)
            player_center_x = max_loc[0] + 20 
            target_col = round((player_center_x + OFFSET_X - 74) / 59.1)
            
            if target_col in range(6):
                cy = 261
                tx_start = int(74+(target_col*59.1))-5
                target_roi = img_n1_gray[cy-5:cy+5, tx_start:tx_start+10]
                
                # Check Target Slot (Must be LIVE ORE)
                if get_slot_state(target_roi, bg_t[0]) == 2:
                    
                    # 3. LEFT-GUTTER NEGATIVE CONSTRAINT
                    # All slots to the left must be PURE GRAVEL (No ores/shadows)
                    left_is_clean = True
                    for l_col in range(target_col):
                        lcx = int(74 + (l_col * 59.1))
                        if get_slot_state(img_n1_gray[cy-5:cy+5, lcx-5:lcx+5], bg_t[0]) > 0:
                            left_is_clean = False
                            break
                    
                    if left_is_clean:
                        # 4. PERSISTENCE CHECK: Only log if DNA has shifted since last call
                        current_dna = [get_slot_state(img_n1_gray[261-5:261+5, int(74+(c*59.1))-5:int(74+(c*59.1))+5], bg_t[0]) for c in range(24)]
                        
                        if current_dna != last_logged_dna:
                            floor_num = len(floor_library) + 1
                            last_logged_dna = current_dna
                            
                            # Log Side-by-Side Handshake
                            bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                            bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                            cv2.putText(bgr_n, f"F{i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                            cv2.putText(bgr_n1, f"F{i+1} (START FLOOR {floor_num})", (30, 50), 0, 0.7, (0,255,0), 2)
                            
                            cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                            floor_library.append({"floor": floor_num, "idx": i+1})
                            print(f" [!] Floor {floor_num} Logged: Frame {i+1} (Col {target_col})")

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v51_master_audit()