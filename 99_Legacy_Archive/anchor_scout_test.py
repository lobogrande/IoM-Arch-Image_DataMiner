import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/MasterAuditor_v95_Pulse"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
MATCH_THRESHOLD = 0.90

def get_stage_hash(img_bgr):
    """Creates a unique hash of the 'Stage: XXX' text box pixels."""
    # ROI for the Stage number area
    roi = img_bgr[65:100, 130:200]
    return hash(roi.tobytes())

def run_v95_pulse_audit():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. BASELINE INITIALIZATION
    floor_library = [{"floor": 1, "idx": 0}]
    img_f1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001.jpg"), img_f1)
    
    last_stage_hash = get_stage_hash(img_f1)
    last_trigger_idx = 0
    
    print(f"--- Running v9.5 Binary-Pulse Auditor ---")

    for i in range(1, len(buffer_files)):
        if i % 500 == 0: 
            print(f" [Scanning] Frame {i} | Floors: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. PLAYER CHECK (Are we at a spawn point?)
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            if any(abs(current_x - a) <= 4 for a in VALID_ANCHORS):
                
                # 3. STAGE VERIFICATION (The "Same-Floor" Killer)
                current_hash = get_stage_hash(img_bgr)
                
                if current_hash != last_stage_hash:
                    # Only trigger if enough time has passed (0.5s / 15 frames)
                    if (i - last_trigger_idx) > 15:
                        floor_num = len(floor_library) + 1
                        last_trigger_idx = i
                        last_stage_hash = current_hash
                        
                        bgr_prev = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i-1]))
                        cv2.putText(bgr_prev, f"F{i-1} (END)", (20, 40), 0, 0.7, (0,0,255), 2)
                        cv2.putText(img_bgr, f"F{i} (START F{floor_num})", (20, 40), 0, 0.7, (0,255,0), 2)
                        
                        cv2.imwrite(os.path.join(OUTPUT_DIR, f"Floor_{floor_num:03}.jpg"), 
                                    np.hstack((bgr_prev, img_bgr)))
                        
                        floor_library.append({"floor": floor_num, "idx": i})
                        print(f"\n [!] TRIGGER: Floor {floor_num} at Frame {i} | Stage Hash Changed")

    with open(f"Run_0_FloorMap_v95.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Mapped {len(floor_library)} floors.")

if __name__ == "__main__":
    run_v95_pulse_audit()