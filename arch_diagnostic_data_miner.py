import cv2
import numpy as np
import os
import json
import csv
import shutil

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. ORE FLOOR RESTRICTIONS ---
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- 3. MASTER CONSTANTS (VERIFIED HUD/ROI) ---
GRID_START_Y, GRID_START_X = 264, 60
CELL_H, CELL_W = 66, 64
HUD_OUT = "Mining_HUD_Analysis"
UNIFIED_ROOT = "Unified_Consensus_Inputs"
# DIAGNOSTIC SENSITIVITY: 0.65 to see "near misses" in Dataset 0
CONFIDENCE_THRESHOLD = 0.65

def get_allowed_tiers(floor):
    allowed = []
    for tier, (start, stop) in ORE_RESTRICTIONS.items():
        if start <= floor <= stop: allowed.append(tier)
    return allowed

def load_templates():
    templates = {}
    t_path = "templates"
    if not os.path.exists(t_path): 
        print(f" [!] ERROR: {t_path} folder not found.")
        return {}
    
    files = [f for f in os.listdir(t_path) if f.endswith('.png')]
    print(f"--- Loading {len(files)} Templates ---")
    
    for f in files:
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        parts = f.split("_")
        if parts[0] in ["background", "xhair"]: continue
        
        tier, state = parts[0], parts[1]
        if tier not in templates: templates[tier] = {'act': [], 'sha': []}
        templates[tier][state].append(img)
    return templates

def run_diagnostic_miner():
    templates = load_templates()
    if not os.path.exists(HUD_OUT): os.makedirs(HUD_OUT)
    global_results = []

    # Verify directory structure
    if not os.path.exists(UNIFIED_ROOT):
        print(f" [!] ERROR: {UNIFIED_ROOT} folder not found.")
        return

    runs = sorted([d for d in os.listdir(UNIFIED_ROOT) if d.startswith("Run_")])
    print(f"Found {len(runs)} Runs to process.")

    for run_dir in runs:
        run_id = run_dir.split("_")[1]
        run_path = os.path.join(UNIFIED_ROOT, run_dir)
        buffer_path = f"capture_buffer_{run_id}"
        json_path = os.path.join(run_path, "final_sequence.json")
        
        if not os.path.exists(json_path): continue
        with open(json_path, 'r') as f: sequence = json.load(f)

        print(f"\n--- DIAGNOSTIC MINING: RUN {run_id} ({len(sequence)} Floors) ---")
        run_hud_dir = os.path.join(HUD_OUT, f"Run_{run_id}")
        if not os.path.exists(run_hud_dir): os.makedirs(run_hud_dir)

        for s_idx, entry in enumerate(sequence):
            floor = entry['floor']
            anchor_idx = entry['idx']
            
            # Safe temporal window (clamped by floor boundaries)
            t_min = sequence[s_idx-1]['idx'] + 2 if s_idx > 0 else 0
            t_max = sequence[s_idx+1]['idx'] - 2 if s_idx < len(sequence)-1 else anchor_idx + 10
            win_start, win_end = max(t_min, anchor_idx - 4), min(t_max, anchor_idx + 4)
            
            allowed = get_allowed_tiers(floor)
            
            # Load window images in grayscale for performance/precision matching
            buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
            window_rois = []
            for t_idx in range(win_start, win_end + 1):
                img = cv2.imread(os.path.join(buffer_path, buffer_files[t_idx]), 0)
                window_rois.append(img)

            # Load Color image for HUD Overlay
            hud_img = cv2.imread(os.path.join(run_path, f"F{floor}_{entry['frame']}"))
            
            # ROI Calibration Check for the first floor of each run
            if floor == 1 or s_idx == 0:
                cal_img = hud_img.copy()
                for slot in range(24):
                    row, col = slot // 6, slot % 6
                    y, x = GRID_START_Y + (row * CELL_H), GRID_START_X + (col * CELL_W)
                    cv2.rectangle(cal_img, (x, y), (x+CELL_W, y+CELL_H), (0, 255, 0), 1)
                cv2.imwrite(os.path.join(run_hud_dir, "Grid_Calibration_Check.jpg"), cal_img)

            for slot in range(24):
                row, col = slot // 6, slot % 6
                y, x = GRID_START_Y + (row * CELL_H), GRID_START_X + (col * CELL_W)
                
                # Boss Floor Truth lookup
                boss_tier = None
                if floor in BOSS_DATA:
                    b = BOSS_DATA[floor]
                    if 'tier' in b and b['tier'] != 'mixed': boss_tier = b['tier']
                    elif 'special' in b and slot in b['special']: boss_tier = b['special'][slot]

                best_match = {'tier': 'empty', 'state': 'none', 'score': 0.0}
                check_list = [boss_tier] if boss_tier else allowed

                for frame_gray in window_rois:
                    roi = frame_gray[y:y+CELL_H, x:x+CELL_W]
                    for tier in check_list:
                        if tier not in templates: continue
                        for state in ['act', 'sha']:
                            for t_img in templates[tier][state]:
                                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                                score = res.max()
                                if score > best_match['score']:
                                    best_match = {'tier': tier, 'state': state, 'score': score}

                if best_match['score'] > CONFIDENCE_THRESHOLD and best_match['tier'] != 'empty':
                    global_results.append({
                        'run': run_id, 'floor': floor, 'slot': slot, 
                        'tier': best_match['tier'], 'state': best_match['state'], 
                        'confidence': f"{best_match['score']:.4f}", 'is_boss': floor in BOSS_DATA
                    })
                    
                    # HUD Overlay with state-color
                    color = (0, 255, 0) if best_match['state'] == 'act' else (0, 165, 255)
                    cv2.rectangle(hud_img, (x+2, y+2), (x+CELL_W-2, y+CELL_H-2), color, 1)
                    cv2.putText(hud_img, f"{best_match['tier']} ({best_match['score']:.2f})", (x+4, y+CELL_H-6), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            cv2.imwrite(os.path.join(run_hud_dir, f"F{floor}_Analysis.jpg"), hud_img)
            if floor % 20 == 0: print(f"  Processed Floor {floor}...")

    # Save to CSV
    with open("archaeology_mining_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'floor', 'slot', 'tier', 'state', 'confidence', 'is_boss'])
        writer.writeheader(); writer.writerows(global_results)
    print(f"\nDiagnostic Complete. Report saved. Identified {len(global_results)} ores.")

if __name__ == "__main__":
    run_diagnostic_miner()