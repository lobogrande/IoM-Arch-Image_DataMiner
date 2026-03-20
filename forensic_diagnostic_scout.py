import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
# Surgical Constants - Loosened for diagnosis
MATCH_THRESHOLD = 0.80 
OFFSET_X = 24           
D_GATE_LIVE = 8.5    
D_GATE_SHADOW = 4.0  

def get_slot_state(roi, bg_template):
    diff = np.sum(cv2.absdiff(roi, bg_template[19:29, 19:29])) / 100
    if diff > D_GATE_LIVE: return 2
    if diff > D_GATE_SHADOW: return 1
    return 0

def run_v43_diagnostic_scout():
    player_right = cv2.imread("templates/player_right.png", 0)
    bg_t = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    frames_since_trigger = 100 

    print(f"--- Running v4.3 Forensic Diagnostic Scout ---")
    print(f"Scanning {len(buffer_files)} frames...")

    for i in range(len(buffer_files) - 1):
        frames_since_trigger += 1
        
        # Periodic Heartbeat
        if i % 500 == 0:
            print(f" [Step] Processing Frame {i}...")

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. SEEK PLAYER
        # Expanded Y-band to 180-380 to ensure we aren't missing a vertical shift
        search_roi = img_n1_gray[180:380, 0:450]
        res = cv2.matchTemplate(search_roi, player_right, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # 2. DIAGNOSTIC LOGGING
        # If we find a "strong-ish" match but it doesn't trigger, find out why
        if max_val > 0.70:
            player_center_x = max_loc[0] + 20 
            target_col = round((player_center_x + OFFSET_X - 74) / 59.1)
            
            if target_col in range(6):
                cy = 261
                tx_start = int(74+(target_col*59.1))-5
                target_roi = img_n1_gray[cy-5:cy+5, tx_start:tx_start+10]
                target_state = get_slot_state(target_roi, bg_t[0])
                
                # Check for shadows to the left
                invalid_left = False
                for left_col in range(target_col):
                    lcx = int(74 + (left_col * 59.1))
                    if get_slot_state(img_n1_gray[cy-5:cy+5, lcx-5:lcx+5], bg_t[0]) > 0:
                        invalid_left = True
                        break
                
                # Log why it almost matched
                if max_val > MATCH_THRESHOLD:
                    if target_state != 2:
                        pass # Valid template, but no ore to the right
                    elif invalid_left:
                        pass # Valid template and ore, but player has already mined to the left
                    elif frames_since_trigger <= 3:
                        pass # Duplicate frame
                    else:
                        # SUCCESSFUL TRIGGER
                        floor_num = len(floor_library) + 1
                        floor_library.append({"floor": floor_num, "idx": i+1})
                        print(f" [!] TRIGGER: Floor {floor_num} at Frame {i+1} (Score: {max_val:.3f} | Col: {target_col})")
                        frames_since_trigger = 0

    print(f"\n[FINISH] Diagnostic complete. Found {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v43_diagnostic_scout()