import cv2
import numpy as np
import os
import json

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOORS = range(1, 11)
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

# MATH GATES
D_GATE = 6      
O_GATE = 0.75   

def get_ui_text_mask(slot_id):
    """Only masks the top row where 'Dig Stage' text is static."""
    mask = np.zeros((48, 48), dtype=np.uint8)
    if slot_id in [1, 2, 3, 4]:
        # Ignore top 18 pixels for UI text
        cv2.rectangle(mask, (5, 18), (43, 45), 255, -1)
    else:
        # Standard central circle for all other slots
        cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def run_temporal_masked_audit():
    bg_templates = [cv2.resize(cv2.imread(os.path.join("templates", f), 0), (48, 48)) 
                    for f in os.listdir("templates") if f.startswith("background")]
    
    ore_templates = []
    for f in os.listdir("templates"):
        if f.startswith("background"): continue
        img = cv2.imread(os.path.join("templates", f), 0)
        if img is not None:
            ore_templates.append({'name': f, 'img': cv2.resize(img, (48, 48))})

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    buffer_path = f"capture_buffer_{TARGET_RUN}"
    buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
    
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- Running v1.4.1 Temporal Masked Audit (Moving Player Support) ---")

    for f_num in TARGET_FLOORS:
        if f_num not in sequence: continue
        anc_idx = sequence[f_num]['idx']
        raw_img = cv2.imread(os.path.join(run_path, f"F{f_num}_{sequence[f_num]['frame']}"))
        gray_main = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        
        for slot in range(24):
            row, col = divmod(slot, 6)
            cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            roi_main = gray_main[y1:y2, x1:x2]

            # 1. OCCUPANCY CHECK
            min_diff = min([np.sum(cv2.absdiff(roi_main, bg)) / (48*48) for bg in bg_templates])
            
            if min_diff <= D_GATE:
                continue # Skip empty slots

            # 2. IDENTIFICATION WITH TEMPORAL RECOVERY
            best_o = 0
            slot_mask = get_ui_text_mask(slot)
            
            # Scan +/- 10 frames to find a moment where the player has moved
            for off in range(-10, 11):
                idx = anc_idx + off
                if not (0 <= idx < len(buffer_files)): continue
                
                # Use ROI from buffer frame
                roi_temp = cv2.imread(os.path.join(buffer_path, buffer_files[idx]), 0)[y1:y2, x1:x2]
                
                for t in ore_templates:
                    res = cv2.matchTemplate(roi_temp, t['img'], cv2.TM_CCORR_NORMED, mask=slot_mask)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if score > best_o: best_o = score
            
            # Final HUD Logic
            color = (0, 255, 0) if best_o > O_GATE else (0, 0, 255)
            label = f"O:{best_o:.2f}" if best_o > O_GATE else f"UNK:{int(min_diff)}"
            
            cv2.rectangle(raw_img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (0,0,0), 2)
            cv2.putText(raw_img, label, (x1+2, y2-4), 0, 0.35, (255,255,255), 1)

        cv2.imwrite(f"Temporal_F{f_num}.jpg", raw_img)
        print(f" [+] Processed Floor {f_num}")

if __name__ == "__main__":
    run_temporal_masked_audit()