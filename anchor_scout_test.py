import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v74_Pulse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def get_row_dna(img_gray):
    """Simple Row 1 DNA for state comparison."""
    vec = []
    for c in range(6):
        cx = int(74 + (c * 59.1))
        roi = img_gray[261-5:261+5, cx-5:cx+5]
        # Using a simple mean to detect 'something vs nothing'
        vec.append(1 if np.mean(roi) > 60 else 0)
    return vec

def run_v74_pulse_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    floor_library = [{"floor": 1, "idx": 0}]
    last_player_at_anchor = False
    last_logged_dna = None

    # Forced Floor 1
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    print(f"--- Running v7.4 State-Pulse Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_gray = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]), 0)
        if img_n1_gray is None: continue

        # 1. SEEK PLAYER
        search_roi = img_n1_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        current_at_anchor = False
        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            if any(abs(current_x - a) <= 4 for a in VALID_ANCHORS):
                current_at_anchor = True

        # 2. THE PULSE TRIGGER
        # If player WAS NOT at an anchor, and NOW IS at an anchor...
        if current_at_anchor and not last_player_at_anchor:
            current_dna = get_row_dna(img_n1_gray)
            
            # 3. DNA PERSISTENCE (Fail-safe against the 55->55 loop)
            if current_dna != last_logged_dna:
                floor_num = len(floor_library) + 1
                last_logged_dna = current_dna
                
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                bgr_n1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
                
                # Visual Labels
                cv2.putText(bgr_n, f"FRAME {i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                cv2.putText(bgr_n1, f"FRAME {i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), np.hstack((bgr_n, bgr_n1)))
                floor_library.append({"floor": floor_num, "idx": i+1})
                
                print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1}")

        last_player_at_anchor = current_at_anchor

    # Export map
    with open(f"Run_0_FloorMap_v74.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v74_pulse_audit()