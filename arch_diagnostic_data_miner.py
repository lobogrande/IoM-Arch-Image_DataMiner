import cv2
import numpy as np
import os
import json
import csv

# --- 1. ORE RESTRICTIONS & BOSS DATA ---
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

# --- 2. THE CALIBRATED CONSTANTS (FROM VERIFICATION REPORT) ---
SLOT1_CENTER = (75, 261) # X, Y
X_STEP, Y_STEP = 59.1, 59.1
AI_W, AI_H = 48, 48

SURGICAL_CONFIG = {
    "target_run": "0",
    "target_floors": [1, 34, 49, 99],
    "confidence_floor": 0.40,
    "output_dir": "Surgical_v1.6_Output"
}

UNIFIED_ROOT = "Unified_Consensus_Inputs"

def load_templates():
    templates = {}
    t_path = "templates"
    files = [f for f in os.listdir(t_path) if f.endswith('.png')]
    for f in files:
        parts = f.split("_")
        if parts[0] in ["background", "xhair"]: continue
        tier, state = parts[0], parts[1]
        if tier not in templates: templates[tier] = {'act': [], 'sha': []}
        templates[tier][state].append(cv2.imread(os.path.join(t_path, f), 0))
    return templates

def run_surgical_v16():
    templates = load_templates()
    run_id = SURGICAL_CONFIG["target_run"]
    run_path = os.path.join(UNIFIED_ROOT, f"Run_{run_id}")
    buffer_path = f"capture_buffer_{run_id}"
    
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    out_dir = SURGICAL_CONFIG["output_dir"]
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])

    for floor in SURGICAL_CONFIG["target_floors"]:
        if floor not in sequence: continue
        entry = sequence[floor]
        print(f"Analyzing Floor {floor}...")
        
        hud_img = cv2.imread(os.path.join(run_path, f"F{floor}_{entry['frame']}"))
        specimen_dir = os.path.join(out_dir, f"F{floor}_Specimens")
        if not os.path.exists(specimen_dir): os.makedirs(specimen_dir)

        for slot in range(24):
            row, col = divmod(slot, 6)
            cx = int(SLOT1_CENTER[0] + (col * X_STEP))
            cy = int(SLOT1_CENTER[1] + (row * Y_STEP))
            
            # ROI Coordinates
            x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
            
            # Save a specimen for visual verification
            specimen = hud_img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(specimen_dir, f"Slot_{slot:02}.jpg"), specimen)

            best = {'tier': 'empty', 'score': 0.0, 'state': 'none'}
            allowed = [t for t, (s, e) in ORE_RESTRICTIONS.items() if s <= floor <= e]
            
            # Temporal Audit (+/- 2 frames)
            for off in range(-2, 3):
                idx = entry['idx'] + off
                if not (0 <= idx < len(buffer_files)): continue
                roi = cv2.imread(os.path.join(buffer_path, buffer_files[idx]), 0)[y1:y2, x1:x2]
                
                for tier in allowed:
                    if tier not in templates: continue
                    for state in ['act', 'sha']:
                        for t_img in templates[tier][state]:
                            # Ensure template and ROI are same size
                            if t_img.shape != (48, 48):
                                t_img = cv2.resize(t_img, (48, 48))
                            
                            res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                            score = res.max()
                            if score > best['score']:
                                best = {'tier': tier, 'score': score, 'state': state}

            if best['score'] > SURGICAL_CONFIG["confidence_floor"]:
                color = (0, 255, 0) if best['state'] == 'act' else (0, 165, 255)
                cv2.rectangle(hud_img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(hud_img, f"{best['tier']}", (x1+2, y2-4), 0, 0.3, color, 1)

        cv2.imwrite(os.path.join(out_dir, f"F{floor}_Analysis.jpg"), hud_img)

if __name__ == "__main__":
    run_surgical_v16()