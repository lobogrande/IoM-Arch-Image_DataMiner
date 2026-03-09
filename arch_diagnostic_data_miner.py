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

# --- 3. SURGICAL CONFIGURATION ---
SURGICAL_CONFIG = {
    "target_run": "0",
    "target_floors": [1, 11, 34, 49, 67, 99], # Add/Remove specific floors here
    "confidence_floor": 0.45,                 # Lowered to capture everything for analysis
    "output_subdir": "Surgical_Tests"
}

# --- 4. MASTER CONSTANTS ---
GRID_START_Y, GRID_START_X = 264, 60
CELL_H, CELL_W = 66, 64
HUD_OUT = "Mining_HUD_Analysis"
UNIFIED_ROOT = "Unified_Consensus_Inputs"

def get_allowed_tiers(floor):
    return [t for t, (s, e) in ORE_RESTRICTIONS.items() if s <= floor <= e]

def load_templates():
    templates = {}
    t_path = "templates"
    if not os.path.exists(t_path): return {}
    
    print(f"--- Loading Templates ---")
    files = [f for f in os.listdir(t_path) if f.endswith('.png')]
    for f in files:
        img = cv2.imread(os.path.join(t_path, f), 0)
        parts = f.split("_")
        if parts[0] in ["background", "xhair"]: continue
        tier, state = parts[0], parts[1]
        if tier not in templates: templates[tier] = {'act': [], 'sha': []}
        templates[tier][state].append(img)
    
    print(f" Manifest: Loaded {len(templates.keys())} ore tiers.")
    return templates

def run_surgical_diagnostic():
    templates = load_templates()
    run_id = SURGICAL_CONFIG["target_run"]
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{run_id}")
    buffer_path = f"capture_buffer_{run_id}"
    
    json_path = os.path.join(run_path, "final_sequence.json")
    if not os.path.exists(json_path):
        print(f"Error: Could not find sequence at {json_path}")
        return

    with open(json_path, 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    # Filter floors
    floors_to_process = [f for f in SURGICAL_CONFIG["target_floors"] if f in sequence]
    print(f"\n--- SURGICAL DIAGNOSTIC: RUN {run_id} | Floors: {floors_to_process} ---")
    
    report = []
    out_dir = os.path.join(HUD_OUT, SURGICAL_CONFIG["output_subdir"])
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])

    for floor in floors_to_process:
        entry = sequence[floor]
        anc_idx = entry['idx']
        
        # Grid Calibration Image (Visual ROI check)
        hud_img = cv2.imread(os.path.join(run_path, f"F{floor}_{entry['frame']}"))
        cal_img = hud_img.copy()
        
        # Determine allowed tiers
        allowed = get_allowed_tiers(floor)
        
        # Scan +/- 3 frames for the surgical audit
        for slot in range(24):
            row, col = slot // 6, slot % 6
            y, x = GRID_START_Y + (row * CELL_H), GRID_START_X + (col * CELL_W)
            cv2.rectangle(cal_img, (x, y), (x+CELL_W, y+CELL_H), (0, 255, 0), 1)

            boss_tier = None
            if floor in BOSS_DATA:
                b = BOSS_DATA[floor]
                if 'tier' in b and b['tier'] != 'mixed': boss_tier = b['tier']
                elif 'special' in b and slot in b['special']: boss_tier = b['special'][slot]
            
            check_list = [boss_tier] if boss_tier else allowed
            best = {'tier': 'empty', 'score': 0.0, 'state': 'none'}

            # Temporal Neighborhood Audit
            for offset in range(-3, 4):
                frame_idx = anc_idx + offset
                if frame_idx < 0 or frame_idx >= len(buffer_files): continue
                
                # Grayscale ROI for matching
                roi = cv2.imread(os.path.join(buffer_path, buffer_files[frame_idx]), 0)[y:y+CELL_H, x:x+CELL_W]
                
                for tier in check_list:
                    if tier not in templates: continue
                    for state in ['act', 'sha']:
                        for t_img in templates[tier][state]:
                            res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                            score = res.max()
                            if score > best['score']:
                                best = {'tier': tier, 'score': score, 'state': state}

            if best['score'] > SURGICAL_CONFIG["confidence_floor"]:
                report.append({'floor': floor, 'slot': slot, 'tier': best['tier'], 'score': f"{best['score']:.4f}", 'state': best['state']})
                color = (0, 255, 0) if best['state'] == 'act' else (0, 165, 255)
                cv2.rectangle(hud_img, (x+2, y+2), (x+CELL_W-2, y+CELL_H-2), color, 1)
                cv2.putText(hud_img, f"{best['tier']} ({best['score']:.2f})", (x+4, y+CELL_H-6), 0, 0.3, color, 1)

        cv2.imwrite(os.path.join(out_dir, f"F{floor}_Surgical_Analysis.jpg"), hud_img)
        cv2.imwrite(os.path.join(out_dir, f"F{floor}_Calibration.jpg"), cal_img)
        print(f" Floor {floor} analyzed. Best Slot Score: {max([float(r['score']) for r in report if r['floor']==floor], default=0):.2f}")

    with open("surgical_mining_diagnostic.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['floor', 'slot', 'tier', 'score', 'state'])
        writer.writeheader(); writer.writerows(report)
    print(f"\nSurgical Audit Complete. Check {out_dir} and surgical_mining_diagnostic.csv")

if __name__ == "__main__":
    run_surgical_diagnostic()