import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/AnchorScout_v42_Surgical_{datetime.now().strftime('%m%d_%H%M')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SURGICAL CONSTANTS
MATCH_THRESHOLD = 0.90 
OFFSET_X = 24           
D_GATE_LIVE = 8.5    # Threshold for Live Ore
D_GATE_SHADOW = 4.0  # Threshold to detect Shadow Templates (lower delta than live)

def get_slot_state(roi, bg_template):
    """Returns: 0=Gravel, 1=Shadow, 2=Live Ore"""
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    if diff > D_GATE_LIVE: return 2
    if diff > D_GATE_SHADOW: return 1
    return 0

def run_v42_surgical_scout():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = []
    frames_since_trigger = 100 

    print(f"--- Running v4.2 High-Speed Surgical Scout ---")

    for i in range(len(buffer_files) - 1):
        frames_since_trigger += 1
        
        if i == 0:
            floor_library.append({"floor": 1, "idx": 0})
            cv2.imwrite(os.path.join(OUTPUT_DIR, "START_Floor001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))
            continue

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. SEEK PLAYER (Full Row 1 Band)
        search_roi = img_n1_gray[230:310, 0:400]
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            # 2. CALCULATE ANCHOR COLUMN
            player_center_x = max_loc[0] + 20 
            target_col = round((player_center_x + OFFSET_X - 74) / 59.1)
            
            if target_col in range(6):
                # 3. SURGICAL SCAN OF ROW 1
                cy = 261
                
                # Check Target Slot (Must be LIVE)
                target_roi = img_n1_gray[cy-5:cy+5, int(74+(target_col*59.1))-5:int(74+(target_col*59.1))+5]
                target_state = get_slot_state(target_roi, bg_t[0])
                
                if target_state == 2: # Target is a Live Ore
                    # 4. NEGATIVE CONSTRAINT: Left slots must be GRAVEL (no shadows allowed)
                    invalid_left = False
                    for left_col in range(target_col):
                        lcx = int(74 + (left_col * 59.1))
                        left_roi = img_n1_gray[cy-5:cy+5, lcx-5:lcx+5]
                        if get_slot_state(left_roi, bg_t[0]) > 0: # 1 (Shadow) or 2 (Live)
                            invalid_left = True
                            break
                    
                    # 5. DYNAMIC TRIGGER
                    # Trigger if (Player at Home) AND (No Ores/Shadows to Left)
                    # Min 3-frame lockout to prevent double-counting the same teleport
                    if not invalid_left and frames_since_trigger > 3:
                        floor_num = len(floor_library) + 1
                        bgr_n, bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i])), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                        
                        cv2.putText(bgr_n, f"F{i} (END)", (30, 50), 0, 0.7, (0,0,255), 2)
                        cv2.putText(bgr_n1, f"F{i+1} (START FLOOR {floor_num})", (30, 50), 0, 0.7, (0,255,0), 2)
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Anchor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                        
                        floor_library.append({"floor": floor_num, "idx": i+1})
                        print(f" [!] Floor {floor_num} Logged: Col {target_col} is First-Live")
                        frames_since_trigger = 0

    print(f"\n[SUCCESS] v4.2 Scout mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v42_surgical_scout()