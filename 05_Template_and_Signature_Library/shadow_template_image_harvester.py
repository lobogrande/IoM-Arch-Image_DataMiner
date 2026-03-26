import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json
import shutil

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. VERIFIED COORDINATES ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48 

# --- 3. CONFIGURATION ---
UNIFIED_ROOT = "Unified_Consensus_Inputs"
OUTPUT_DIR = "Standardized_Templates_Raw"
TARGET_RUN = "0"
# Starting with your requested Boss Floors
TARGET_FLOORS = [2, 3, 4, 5, 6, 9]

def run_interactive_harvester():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    buffer_path = f"capture_buffer_{TARGET_RUN}"
    
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}
    
    buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
    
    cv2.namedWindow("Shadow Harvester Preview", cv2.WINDOW_NORMAL)
    
    print("\n--- SHADOW HARVESTER (v2.2) ---")
    print("WASD: Adjust Offset | ENTER: Extract All Slots | SPACE: Next Floor | ESC: Exit")

    for floor in TARGET_FLOORS:
        if floor not in sequence: continue
        
        anchor_idx = sequence[floor]['idx']
        offset = 0
        
        while True:
            target_idx = np.clip(anchor_idx + offset, 0, len(buffer_files)-1)
            frame_path = os.path.join(buffer_path, buffer_files[target_idx])
            img = cv2.imread(frame_path)
            if img is None: break
            
            # Draw preview indicators (the 48x48 search areas)
            display_img = img.copy()
            for slot in range(24):
                row, col = divmod(slot, 6)
                cx = int(SLOT1_CENTER[0] + (col * STEP_X))
                cy = int(SLOT1_CENTER[1] + (row * STEP_Y))
                cv2.rectangle(display_img, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 255), 1)

            cv2.putText(display_img, f"Floor: {floor} | Offset: {offset} | Frame: {buffer_files[target_idx]}", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Shadow Harvester Preview", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d'): offset += 1
            elif key == ord('a'): offset -= 1
            elif key == ord('w'): offset += 10
            elif key == ord('s'): offset -= 10
            
            elif key == 13: # ENTER: Harvest
                floor_out = os.path.join(OUTPUT_DIR, f"Floor_{floor}_Off_{offset}")
                if not os.path.exists(floor_out): os.makedirs(floor_out)
                
                for slot in range(24):
                    row, col = divmod(slot, 6)
                    cx = int(SLOT1_CENTER[0] + (col * STEP_X))
                    cy = int(SLOT1_CENTER[1] + (row * STEP_Y))
                    crop = img[cy-24:cy+24, cx-24:cx+24]
                    
                    # Boss Hint Filename
                    hint = "unknown"
                    if floor in cfg.BOSS_DATA:
                        b = cfg.BOSS_DATA[floor]
                        if 'tier' in b and b['tier'] != 'mixed': hint = b['tier']
                        elif 'special' in b and slot in b['special']: hint = b['special'][slot]
                    
                    fname = f"S{slot:02}_F{floor}_O{offset}_{hint}.png"
                    cv2.imwrite(os.path.join(floor_out, fname), crop)
                
                print(f" [!] Successfully harvested 24 templates from Floor {floor} (Offset {offset})")

            elif key == 32 or key == ord('n'): # SPACE / N: Next Floor
                break
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("\n[SUCCESS] Finished target floor list.")

if __name__ == "__main__":
    run_interactive_harvester()