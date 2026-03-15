import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/FirstStrike_v10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THE 6 VERIFIED GREEN-LINE ANCHORS
# Player X positions matching Slot 0, 1, 2, 3, 4, 5
VALID_ANCHORS = [11, 70, 129, 188, 247, 306]
# Center X of Ores for "Shadow Check" (Slot 0, 1, 2, 3, 4, 5)
ORE_SLOTS_X = [74, 133, 192, 251, 310, 369] 
ORE_Y = 261 # Center of Row 1

MATCH_THRESHOLD = 0.92

def is_slot_occupied(img_gray, slot_idx):
    """Detects if any ore (shadow or active) exists in a specific slot."""
    cx = ORE_SLOTS_X[slot_idx]
    # Sample a 14x14 box at the center of the slot
    roi = img_gray[ORE_Y-7:ORE_Y+7, cx-7:cx+7]
    # If the average brightness is high, an ore is present
    return np.mean(roi) > 60 

def run_v10_first_strike():
    player_t = cv2.imread("templates/player_right.png", 0)
    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # 1. INITIALIZE FLOOR 1
    floor_library = [{"floor": 1, "idx": 0, "anchor_slot": 0}]
    img_f1 = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0]))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Floor_001_Frame_00000.jpg"), img_f1)

    print(f"--- Running v10.0 First-Strike Auditor ---")
    
    last_logged_idx = 0

    for i in range(1, len(buffer_files)):
        # Terminal Tracking
        if i % 1000 == 0: 
            print(f" [Scanning] Frame {i:05} | Floors Mapped: {len(floor_library)}", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # STEP 1: MATCH PLAYER ONLY IN TOP ROW
        # ROI focused on the Row 1 player belt
        search_roi = img_gray[200:350, 0:480]
        res = cv2.matchTemplate(search_roi, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > MATCH_THRESHOLD:
            current_x = max_loc[0]
            
            # STEP 2: FIND WHICH SLOT PLAYER IS ANCHORED TO
            anchor_slot = None
            for idx, a in enumerate(VALID_ANCHORS):
                if abs(current_x - a) <= 3:
                    anchor_slot = idx
                    break
            
            if anchor_slot is not None:
                # STEP 3: THE "LOOK LEFT" SHADOW CHECK
                # If we are at Slot 2, check Slots 0 and 1.
                is_first_frame = True
                for s in range(anchor_slot):
                    if is_slot_occupied(img_gray, s):
                        is_first_frame = False
                        break
                
                if is_first_frame:
                    # COOLDOWN: Prevent logging the same teleport window multiple times
                    if (i - last_logged_idx) > 20:
                        floor_num = len(floor_library) + 1
                        last_logged_idx = i
                        
                        # Output Image with Outline
                        cv2.rectangle(img_bgr, (max_loc[0], max_loc[1]+200), 
                                      (max_loc[0]+40, max_loc[1]+260), (0, 0, 255), 2)
                        
                        filename = f"Floor_{floor_num:03}_Frame_{i:05}.jpg"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img_bgr)
                        
                        floor_library.append({
                            "floor": floor_num, 
                            "idx": i, 
                            "anchor_slot": anchor_slot
                        })
                        
                        print(f"\n [!] NEW FLOOR: {floor_num} | Frame: {i} | Slot: {anchor_slot}")

    # FINAL EXPORT
    with open(f"Run_0_FloorMap_v10.json", 'w') as f:
        json.dump(floor_library, f, indent=4)

    print(f"\n[FINISH] Analysis complete. {len(floor_library)} floors mapped.")

if __name__ == "__main__":
    run_v10_first_strike()