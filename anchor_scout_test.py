import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v85_Arrival"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def run_v85_arrival_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. INITIALIZE DATASET
    floor_library = [{"floor": 1, "idx": 0}]
    img_start = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), img_start)

    last_at_anchor = True # Assume true for F1
    last_trigger_idx = 0
    
    print(f"--- Running v8.5 Arrival Auditor ---")

    for i in range(len(buffer_files) - 1):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_n1_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
        if img_n1_bgr is None: continue
        img_n1_gray = cv2.cvtColor(img_n1_bgr, cv2.COLOR_BGR2GRAY)

        # 2. SEEK PLAYER
        search_roi = img_n1_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        current_at_anchor = False
        if max_val > MATCH_THRESHOLD:
            # Check if current X matches a verified spawn anchor
            if any(abs(max_loc[0] - a) <= 4 for a in VALID_ANCHORS):
                current_at_anchor = True

        # 3. THE ARRIVAL TRIGGER
        # We fire only if we just arrived at an anchor from a non-anchor state
        if current_at_anchor and not last_at_anchor:
            # Protection: Minimum 15 frames per floor (approx 0.5s)
            if (i - last_trigger_idx) > 15:
                floor_num = len(floor_library) + 1
                last_trigger_idx = i
                
                bgr_n = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
                
                # Visual Labels for Audit
                cv2.putText(bgr_n, f"F{i} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                cv2.putText(img_n1_bgr, f"F{i+1} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                
                out_path = os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg")
                cv2.imwrite(out_path, np.hstack((bgr_n, img_n1_bgr)))
                
                floor_library.append({"floor": floor_num, "idx": i+1})
                print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i+1} | Pos: {max_loc[0]}")

        # Update state for next frame
        last_at_anchor = current_at_anchor

    # Save Final JSON
    with open(f"Run_0_FloorMap_v85.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v85_arrival_audit()