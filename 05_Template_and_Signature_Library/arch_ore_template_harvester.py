import cv2
import numpy as np
import os
import json
import shutil

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. YOUR VERIFIED COORDINATES ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48 # Resulting 48x48 crops

# --- 3. HARVEST CONFIG ---
HARVEST_CONFIG = {
    "target_run": "0",
#    "target_floors": [11,17,23,25,29,31,34,35,41,44,49,74,98,99], # Best variety for templates
    "target_floors": [15,16,19,37,39,104,105,106,108], # Best variety for templates
    "output_dir": "Standardized_Templates_Raw"
}

UNIFIED_ROOT = "Unified_Consensus_Inputs"

def run_harvester_v11():
    out_root = HARVEST_CONFIG["output_dir"]
    if not os.path.exists(out_root): os.makedirs(out_root)
    
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{HARVEST_CONFIG['target_run']}")
    json_path = os.path.join(run_path, "final_sequence.json")
    
    with open(json_path, 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    print(f"--- HARVESTING TEMPLATES WITH ALIGNMENT (74, 261) ---")

    for floor in HARVEST_CONFIG["target_floors"]:
        if floor not in sequence: continue
        entry = sequence[floor]
        img = cv2.imread(os.path.join(run_path, f"F{floor}_{entry['frame']}"))
        
        floor_dir = os.path.join(out_root, f"Floor_{floor}")
        if os.path.exists(floor_dir): shutil.rmtree(floor_dir)
        os.makedirs(floor_dir)

        for slot in range(24):
            row, col = divmod(slot, 6)
            cx = int(SLOT1_CENTER[0] + (col * STEP_X))
            cy = int(SLOT1_CENTER[1] + (row * STEP_Y))
            
            # Crop 48x48 centered on the tuner's crosshair
            x1, y1 = cx - 24, cy - 24
            x2, y2 = cx + 24, cy + 24
            crop = img[y1:y2, x1:x2]
            
            # Boss Hint logic
            hint = "unknown"
            if floor in BOSS_DATA:
                b = BOSS_DATA[floor]
                if 'tier' in b and b['tier'] != 'mixed': hint = b['tier']
                elif 'special' in b and slot in b['special']: hint = b['special'][slot]
            
            fname = f"S{slot:02}_F{floor}_{hint}.png"
            cv2.imwrite(os.path.join(floor_dir, fname), crop)
        
        print(f" [+] Harvested Floor {floor} to {floor_dir}")

if __name__ == "__main__":
    run_harvester_v11()