import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os
import json
import csv

# --- 1. ORE RESTRICTIONS & BOSS DATA ---
# cfg.ORE_RESTRICTIONS moved to project_config

# cfg.BOSS_DATA moved to project_config

UNIFIED_ROOT = "Unified_Consensus_Inputs"

def load_templates():
    templates = {}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return {}
    files = [f for f in os.listdir(t_path) if f.endswith('.png')]
    for f in files:
        parts = f.split("_")
        if parts[0] in ["background", "xhair"]: continue
        tier, state = parts[0], parts[1]
        if tier not in templates: templates[tier] = {'act': [], 'sha': []}
        t_img = cv2.imread(os.path.join(t_path, f), 0)
        # Ensure templates match AI_DIM
        if t_img.shape != (AI_DIM, AI_DIM):
            t_img = cv2.resize(t_img, (AI_DIM, AI_DIM))
        templates[tier][state].append(t_img)
    return templates

def run_calibrated_miner():
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
        print(f"Mining Floor {floor} with v1.7 Calibration...")
        
        hud_img = cv2.imread(os.path.join(run_path, f"F{floor}_{entry['frame']}"))
        specimen_dir = os.path.join(out_dir, f"F{floor}_Centered_Specimens")
        if not os.path.exists(specimen_dir): os.makedirs(specimen_dir)

        for slot in range(24):
            row, col = divmod(slot, 6)
            cx = int(SLOT1_CENTER[0] + (col * X_STEP))
            cy = int(SLOT1_CENTER[1] + (row * Y_STEP))
            
            x1, y1 = cx - (AI_DIM//2), cy - (AI_DIM//2)
            x2, y2 = x1 + AI_DIM, y1 + AI_DIM
            
            # Diagnostic Specimen with Alignment Crosshair
            spec_crop = hud_img[y1:y2, x1:x2].copy()
            cv2.line(spec_crop, (AI_DIM//2, 0), (AI_DIM//2, AI_DIM), (255,0,0), 1) # Blue crosshair
            cv2.line(spec_crop, (0, AI_DIM//2), (AI_DIM, AI_DIM//2), (255,0,0), 1)
            cv2.imwrite(os.path.join(specimen_dir, f"Slot_{slot:02}.jpg"), spec_crop)

            best = {'tier': 'empty', 'score': 0.0, 'state': 'none'}
            allowed = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= floor <= e]
            
            # Check cfg.BOSS_DATA logic
            boss_tier = None
            if floor in cfg.BOSS_DATA:
                b = cfg.BOSS_DATA[floor]
                if 'tier' in b and b['tier'] != 'mixed': boss_tier = b['tier']
                elif 'special' in b and slot in b['special']: boss_tier = b['special'][slot]
            
            check_list = [boss_tier] if boss_tier else allowed

            # Temporal Search (+/- 3 frames)
            for off in range(-3, 4):
                idx = entry['idx'] + off
                if not (0 <= idx < len(buffer_files)): continue
                roi = cv2.imread(os.path.join(buffer_path, buffer_files[idx]), 0)[y1:y2, x1:x2]
                
                for tier in check_list:
                    if tier not in templates: continue
                    for state in ['act', 'sha']:
                        for t_img in templates[tier][state]:
                            res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED)
                            if res.max() > best['score']:
                                best = {'tier': tier, 'score': res.max(), 'state': state}

            if best['score'] > SURGICAL_CONFIG["confidence_floor"]:
                color = (0, 255, 0) if best['state'] == 'act' else (0, 165, 255)
                cv2.rectangle(hud_img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(hud_img, f"{best['tier']} ({best['score']:.2f})", (x1, y1-2), 0, 0.3, color, 1)

        cv2.imwrite(os.path.join(out_dir, f"F{floor}_Mining_Analysis.jpg"), hud_img)

if __name__ == "__main__":
    run_calibrated_miner()