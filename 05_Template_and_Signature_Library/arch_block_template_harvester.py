import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json
import shutil

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
# cfg.BOSS_DATA moved to project_config

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
            if floor in cfg.BOSS_DATA:
                b = cfg.BOSS_DATA[floor]
                if 'tier' in b and b['tier'] != 'mixed': hint = b['tier']
                elif 'special' in b and slot in b['special']: hint = b['special'][slot]
            
            fname = f"S{slot:02}_F{floor}_{hint}.png"
            cv2.imwrite(os.path.join(floor_dir, fname), crop)
        
        print(f" [+] Harvested Floor {floor} to {floor_dir}")

if __name__ == "__main__":
    run_harvester_v11()