import cv2
import numpy as np
import os
import csv
import shutil
from collections import Counter

# --- CONFIGURATION ---
BUFFER_DIR = "capture_buffer"
AUDIT_RESULTS = "audit_verification"
STILLS_DIR = "verification_stills"
DIV_FOLDER = "divine_receipts"
CSV_FILE = "high_fidelity_v29_results.csv"

START_FLOOR = 79 
HEADER_Y, HEADER_X, HEADER_H, HEADER_W = 56, 100, 16, 35

# Slot Coords (With +7px Y-offset for bottom row)
SLOT_COORDS = [(x-10, y-225-5) for x, y in [
    (82, 492), (141, 492), (200, 492), (259, 492), (318, 492), (377, 492),
    (82, 551), (141, 551), (200, 551), (259, 551), (318, 551), (377, 551),
    (82, 610), (141, 610), (200, 610), (259, 610), (318, 610), (377, 610),
    (82, 664+7), (141, 664+7), (200, 664+7), (259, 664+7), (318, 664+7), (377, 664+7)
]]

# Spawn Gates (Strict Assertions)
SPAWN_GATES = {
    'dirt1': (1, 11),   'com1': (1, 17),   'rare1': (3, 25),  'epic1': (6, 29),
    'leg1': (12, 31),   'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23),  'com2': (18, 28),  'rare2': (26, 35), 'epic2': (30, 41),
    'leg2': (32, 44),   'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999),
    'leg3': (45, 999),  'myth3': (50, 999), 'div3': (100, 999)
}

# Precision Tuning
SHA_THRESH = 0.70
VOID_THRESH = 0.82
PLAIN_THRESH = 0.75 
MOD_THRESH = 0.60
DIGIT_THRESH = 0.60

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))

# --- LOADERS ---
def load_all_templates():
    templates = {'act': {'plain': {}, 'mod': {}}, 'sha': {}, 'void': [], 'digits': []}
    if os.path.exists("digits"):
        for f in sorted(os.listdir("digits")):
            if f.endswith(".png"):
                img = cv2.imread(f"digits/{f}", 0)
                if img is not None: templates['digits'].append((int(f[0]), img))
    if os.path.exists("templates"):
        for f in os.listdir("templates"):
            if not f.endswith(".png"): continue
            img = cv2.imread(f"templates/{f}", 0)
            if any(k in f for k in ["background", "fairy", "bone"]):
                templates['void'].append(img); continue
            parts = f.replace('.png', '').split('_')
            tier, state = parts[0], parts[1]
            if state == 'sha':
                if tier not in templates['sha']: templates['sha'][tier] = []
                templates['sha'][tier].append(img)
            else:
                cat = 'mod' if any(m in f for m in ['pmod', 'xhair', 'sprite', 'hbar', 'play']) else 'plain'
                if tier not in templates['act'][cat]: templates['act'][cat][tier] = []
                templates['act'][cat][tier].append(img)
    return templates

def get_header_num(gray_roi, digit_temps):
    _, bin_img = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)
    matches = []
    for val, temp in digit_temps:
        res = cv2.matchTemplate(bin_img, temp, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= DIGIT_THRESH)
        for pt in zip(*locs[::-1]):
            matches.append({'x': pt[0], 'val': val})
    matches.sort(key=lambda d: d['x'])
    unique = []
    if matches:
        unique.append(matches[0])
        for i in range(1, len(matches)):
            if matches[i]['x'] - unique[-1]['x'] >= 4:
                unique.append(matches[i])
    if not unique: return None
    if len(unique) == 1: return unique[0]['val']
    if len(unique) == 2: return unique[0]['val'] * 10 + unique[1]['val']
    if len(unique) == 3: return unique[0]['val'] * 100 + unique[1]['val'] * 10 + unique[2]['val']
    return None

def run_v29_audit():
    # --- CLEANUP OLD RESULTS ---
    for d in [AUDIT_RESULTS, STILLS_DIR, DIV_FOLDER]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    temps = load_all_templates()
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(".png")])
    if not frames: 
        print(f"Error: No frames found in {BUFFER_DIR}")
        return

    cur_floor = START_FLOOR
    best_slots = [{'tier': None, 'score': 0, 'f_name': None} for _ in range(24)]
    has_peaked_shadows = False
    last_transition_idx = 0
    
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Floor", "Slot", "Tier", "Score"])

        print(f"Starting v29.0 Clean Audit: {len(frames)} frames...")

        for idx, f_name in enumerate(frames):
            img_path = os.path.join(BUFFER_DIR, f_name)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 1. State Census (Track Shadows)
            current_shadows = 0
            for s_id, (cx, cy) in enumerate(SLOT_COORDS):
                cell = gray[max(0, cy-30):cy+30, max(0, cx-30):cx+30]
                is_sha = False
                for t_list in temps['sha'].values():
                    for t in t_list:
                        if cv2.matchTemplate(cell, t, cv2.TM_CCOEFF_NORMED).max() > SHA_THRESH:
                            is_sha = True; break
                    if is_sha: break
                if is_sha: current_shadows += 1

            if current_shadows >= 12: has_peaked_shadows = True
            
            # 2. TRANSITION TRIGGER (Shadow Peak followed by Reset)
            if has_peaked_shadows and current_shadows <= 3 and (idx - last_transition_idx > 40):
                header_roi = gray[HEADER_Y:HEADER_Y+HEADER_H, HEADER_X:HEADER_X+HEADER_W]
                ocr_num = get_header_num(header_roi, temps['digits'])
                
                prev_floor = cur_floor
                # UNIDIRECTIONAL GUARDRAIL
                if ocr_num and ocr_num > cur_floor:
                    cur_floor = ocr_num
                else:
                    cur_floor += 1

                tally = [s['tier'] for s in best_slots if s['tier']]
                print(f"-> STAGE {prev_floor} DONE: {dict(Counter(tally))} (Transition at {f_name})")
                
                # Create Census Summary Image
                valid_f_names = [s['f_name'] for s in best_slots if s['f_name']]
                census_f = valid_f_names[0] if valid_f_names else f_name
                census_img = cv2.imread(os.path.join(BUFFER_DIR, census_f))
                
                if census_img is not None:
                    for s_id, data in enumerate(best_slots):
                        if data['tier']:
                            cx, cy = SLOT_COORDS[s_id]
                            # Draw shadow + text for high contrast visibility
                            cv2.putText(census_img, data['tier'], (cx-24, cy+1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                            cv2.putText(census_img, data['tier'], (cx-25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    cv2.imwrite(f"{AUDIT_RESULTS}/Census_F{prev_floor}_idx{idx}.png", census_img)

                # Save Transition Stills (Exit and Entrance)
                t_folder = os.path.join(STILLS_DIR, f"Break_F{prev_floor}_to_F{cur_floor}")
                os.makedirs(t_folder, exist_ok=True)
                if idx > 0:
                    shutil.copy(os.path.join(BUFFER_DIR, frames[idx-1]), os.path.join(t_folder, f"EXIT_F{prev_floor}.png"))
                shutil.copy(img_path, os.path.join(t_folder, f"ENTRANCE_F{cur_floor}.png"))

                # RESET for new floor
                best_slots = [{'tier': None, 'score': 0, 'f_name': None} for _ in range(24)]
                has_peaked_shadows = False
                last_transition_idx = idx
                print(f"NEW STAGE DETECTED: {cur_floor}")

            # 3. TEMPORAL ACCUMULATION (Scan every 4th frame for high-confidence IDs)
            if idx % 4 == 0:
                for s_id, (cx, cy) in enumerate(SLOT_COORDS):
                    cell = gray[max(0, cy-30):cy+30, max(0, cx-30):cx+30]
                    
                    # --- PRIORITY VOID CHECK ---
                    is_void = False
                    for v_temp in temps['void']:
                        if cv2.matchTemplate(cell, v_temp, cv2.TM_CCOEFF_NORMED).max() > VOID_THRESH:
                            is_void = True; break
                    if is_void: continue
                    
                    # --- ORE IDENTIFICATION ---
                    p_cell = clahe.apply(cell)
                    best_m, max_v = None, 0
                    for cat in ['plain', 'mod']:
                        for tier, imgs in temps['act'][cat].items():
                            if tier in SPAWN_GATES and not (SPAWN_GATES[tier][0] <= cur_floor <= SPAWN_GATES[tier][1]): 
                                continue
                            for t in imgs:
                                score = cv2.matchTemplate(p_cell, t, cv2.TM_CCOEFF_NORMED).max()
                                thresh = PLAIN_THRESH if cat == 'plain' else MOD_THRESH
                                if score > thresh and score > max_v:
                                    max_v, best_m = score, tier
                    
                    if best_m and max_v > best_slots[s_id]['score']:
                        best_slots[s_id] = {'tier': best_m, 'score': max_v, 'f_name': f_name}
                        writer.writerow([f_name, cur_floor, s_id, best_m, f"{max_v:.2f}"])
                        if 'div' in best_m:
                            cv2.imwrite(f"{DIV_FOLDER}/{best_m}_F{cur_floor}_{f_name}.png", img_bgr)

            if idx % 500 == 0: 
                print(f"Progress: {idx}/{len(frames)} frames analyzed...")

    print("\nAUDIT COMPLETE.")

if __name__ == "__main__":
    run_v29_audit()