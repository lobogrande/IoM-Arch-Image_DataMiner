import cv2
import numpy as np
import os
import json
import csv
import pandas as pd

# --- 1. MASTER BOSS DATA ---
BOSS_DATA = {11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'}, 25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'}, 34: {'tier': 'mixed', 'special': {0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2', 6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2', 12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2', 18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'}}, 35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'}, 49: {"tier": "mixed", "special": {0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3", 6: "com3", 7: "com3", 8: "com3", 9: "com3", 10: "com3", 11: "com3", 12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3", 18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"}}, 74: {'tier': 'mixed', 'special': {0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3', 6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3', 12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3', 18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'}}, 98: {'tier': 'myth3'}, 99: {"tier": "mixed", "special": {0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2", 6: "com3", 7: "rare3", 8: "epic3", 9: "leg3", 10: "myth3", 11: "div2", 12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2", 18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"}}}

ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

# --- 2. CALIBRATED COORDINATES ---
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1
AI_DIM = 48
UNIFIED_ROOT = "Unified_Consensus_Inputs"
MINING_OUT = "Production_Mining_Results"
FINAL_CSV = "archaeology_final_mining_data.csv"

def load_all_templates():
    templates = {'ore': {}, 'bg': []}
    t_path = "templates"
    if not os.path.exists(t_path): return templates
    files = [f for f in os.listdir(t_path) if f.endswith('.png')]
    for f in files:
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        if img.shape != (AI_DIM, AI_DIM): img = cv2.resize(img, (AI_DIM, AI_DIM))
        if f.startswith("background"):
            templates['bg'].append(img)
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in templates['ore']: templates['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: templates['ore'][tier][state].append(img)
    return templates

def perform_mining_pass(templates):
    print("\n[PHASE 1] Initializing Production Mining Suite (v3.2)...")
    global_report = []
    runs = sorted([d for d in os.listdir(UNIFIED_ROOT) if d.startswith("Run_")])
    
    for run_dir in runs:
        run_id = run_dir.split("_")[1]
        run_path = os.path.join(UNIFIED_ROOT, run_dir)
        buffer_path = f"capture_buffer_{run_id}"
        with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
            sequence = json.load(f)

        print(f"\n>>> PROCESSING RUN {run_id} ({len(sequence)} Floors) <<<")
        hud_dir = os.path.join(MINING_OUT, f"Run_{run_id}_HUD")
        if not os.path.exists(hud_dir): os.makedirs(hud_dir)
        buffer_files = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        
        # Sensitivity Adjustment
        base_threshold = 0.74 if run_id == "3" else 0.77

        for entry in sequence:
            floor, anc_idx = entry['floor'], entry['idx']
            hud_img = cv2.imread(os.path.join(run_path, f"F{floor}_{entry['frame']}"))
            allowed = [t for t, (s, e) in ORE_RESTRICTIONS.items() if s <= floor <= e]
            
            for slot in range(24):
                row, col = divmod(slot, 6)
                cx, cy = int(SLOT1_CENTER[0] + (col * STEP_X)), int(SLOT1_CENTER[1] + (row * STEP_Y))
                x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24

                boss_tier = None
                if floor in BOSS_DATA:
                    b = BOSS_DATA[floor]
                    if 'tier' in b and b['tier'] != 'mixed': boss_tier = b['tier']
                    elif 'special' in b and slot in b['special']: boss_tier = b['special'][slot]
                
                check_list = [boss_tier] if boss_tier else allowed
                best_ore = {'tier': 'empty', 'score': 0.0, 'state': 'none'}
                best_bg_score = 1.0 # Background score starts at 1.0 (perfect match)

                # --- SUB-STEP A: Specialist Scan (Standard +/- 3) ---
                for off in range(-3, 4):
                    idx = anc_idx + off
                    if not (0 <= idx < len(buffer_files)): continue
                    roi = cv2.imread(os.path.join(buffer_path, buffer_files[idx]), 0)[y1:y2, x1:x2]
                    
                    for bg_img in templates['bg']:
                        bg_res = cv2.matchTemplate(roi, bg_img, cv2.TM_CCOEFF_NORMED).max()
                        best_bg_score = min(best_bg_score, bg_res) # Track lowest correlation to BG

                    for tier in check_list:
                        if tier not in templates['ore']: continue
                        for state in ['act', 'sha']:
                            for t_img in templates['ore'][tier][state]:
                                res = cv2.matchTemplate(roi, t_img, cv2.TM_CCOEFF_NORMED).max()
                                if state == 'sha': res *= 1.03
                                if res > best_ore['score']:
                                    best_ore = {'tier': tier, 'score': res, 'state': state}

                # --- SUB-STEP B: Temporal Recovery (Expansion +/- 15) ---
                # Triggered if no ID found but slot doesn't look like background
                is_identified = best_ore['score'] > base_threshold and (best_ore['score'] - (1.0 - best_bg_score) > 0.02)
                
                if not is_identified and best_bg_score < 0.65:
                    for off in range(-15, 16):
                        idx = anc_idx + off
                        if not (0 <= idx < len(buffer_files)): continue
                        roi_deep = cv2.imread(os.path.join(buffer_path, buffer_files[idx]), 0)[y1:y2, x1:x2]
                        
                        # Re-scan specialist templates in wider window to bypass Fairy
                        for tier in check_list:
                            if tier not in templates['ore']: continue
                            for state in ['act', 'sha']:
                                for t_img in templates['ore'][tier][state]:
                                    res = cv2.matchTemplate(roi_deep, t_img, cv2.TM_CCOEFF_NORMED).max()
                                    if state == 'sha': res *= 1.03
                                    if res > best_ore['score']:
                                        best_ore = {'tier': tier, 'score': res, 'state': state}
                    
                    # Final re-check after deep scan
                    is_identified = best_ore['score'] > base_threshold

                # --- HUD DRAWING & REPORTING ---
                if is_identified:
                    status = best_ore['tier']
                    color = (0, 255, 0) if best_ore['state'] == 'act' else (0, 165, 255)
                elif best_bg_score < 0.65:
                    status = "UNKNOWN"
                    color = (0, 0, 255) # Red for unidentified occupancy
                else:
                    continue # Truly empty

                global_report.append({'run': run_id, 'floor': floor, 'slot': slot, 'tier': status, 'state': best_ore['state'], 'score': f"{best_ore['score']:.4f}"})
                
                cv2.rectangle(hud_img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(hud_img, status, (x1+2, y2-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            cv2.imwrite(os.path.join(hud_dir, f"F{floor}_Overlay.jpg"), hud_img)
            if floor % 10 == 0:
                print(f"  [Progress] Floor {floor} complete...")

    with open(FINAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'floor', 'slot', 'tier', 'state', 'score'])
        writer.writeheader(); writer.writerows(global_report)

def run_truth_audit():
    print("\n[PHASE 2] Starting Boss Truth Audit...")
    df = pd.read_csv(FINAL_CSV)
    results = []
    total_checks, correct_calls = 0, 0

    for floor, boss_info in BOSS_DATA.items():
        expected = boss_info['special'] if boss_info.get('tier') == 'mixed' else {i: boss_info['tier'] for i in range(24)}
        for run_id in df['run'].unique():
            actual_map = {row['slot']: row['tier'] for _, row in df[(df['run'] == int(run_id)) & (df['floor'] == floor)].iterrows()}
            for slot in range(24):
                total_checks += 1
                exp, act = expected.get(slot), actual_map.get(slot)
                if exp == act: correct_calls += 1
                else: results.append({'run': run_id, 'floor': floor, 'slot': slot, 'expected': exp or "empty", 'actual': act or "empty"})

    accuracy = (correct_calls / total_checks) * 100
    print(f"\n==========================================\n FINAL BOSS TRUTH ACCURACY: {accuracy:.2f}%\n==========================================")
    if results: pd.DataFrame(results).to_csv("mining_error_log.csv", index=False)

if __name__ == "__main__":
    templates = load_all_templates()
    perform_mining_pass(templates)
    run_truth_audit()