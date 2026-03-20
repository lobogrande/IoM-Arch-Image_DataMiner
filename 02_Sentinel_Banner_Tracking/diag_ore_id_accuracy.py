# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using DNA-Aligned ROI.
# Version: 4.6 (Threshold Adoption & Recovery Verification)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "recovery_verification")

# ROI CONSTANTS
AI_DIM = 30  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# ADOPTED OPTIMIZED THRESHOLDS (Based on v4.5 Waterfall Analysis)
BG_OCCUPANCY_FLOOR = 0.82  
ORE_STRICT_GATE = 0.42     # Recovering ores with lower absolute intensity (catches the 0.49 cluster)
MARGIN_REQUIREMENT = 0.07  # Capturing the high-separation ores
SHADOW_BOOST = 1.04        # Compensating for Row 4 depth/shadows

def load_all_templates():
    templates = {'ore': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        
        h, w = img.shape
        if h > AI_DIM or w > AI_DIM:
            cy, cx = h // 2, w // 2
            r = AI_DIM // 2
            img = img[cy-r : cy+r, cx-r : cx+r]
        elif h < AI_DIM or w < AI_DIM:
            img = cv2.resize(img, (AI_DIM, AI_DIM))
            
        if f.startswith("background") or f.startswith("negative_ui"):
            templates['bg'].append({'id': f, 'img': img})
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            if tier not in templates['ore']: templates['ore'][tier] = {'act': [], 'sha': []}
            if state in ['act', 'sha']: templates['ore'][tier][state].append({'id': f, 'img': img})
    return templates

def process_single_frame(frame_data, templates, buffer_dir):
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return []
    
    row4_y = int(ORE0_Y + (3 * STEP))
    frame_results = []
    recovery_detected = False

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        x1, y1 = int(cx - (AI_DIM//2) - JITTER), int(row4_y - (AI_DIM//2) - JITTER)
        search_area = img_gray[y1 : y1 + AI_DIM + (JITTER*2), x1 : x1 + AI_DIM + (JITTER*2)]
        if search_area.shape[0] < AI_DIM or search_area.shape[1] < AI_DIM: continue

        # 1. Background Defense
        best_bg = {'id': 'none', 'score': 0.0}
        for bg_tpl in templates['bg']:
            res = cv2.matchTemplate(search_area, bg_tpl['img'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best_bg['score']: best_bg = {'id': bg_tpl['id'], 'score': val}

        # 2. Ore Identification
        best_ore = {'tier': 'empty', 'score': 0.0, 'id': 'none'}
        for tier, states in templates['ore'].items():
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_area, ore_tpl['img'], cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if state == 'sha': score *= SHADOW_BOOST
                    if score > best_ore['score']:
                        best_ore = {'tier': tier, 'score': score, 'id': ore_tpl['id']}

        margin = best_ore['score'] - best_bg['score']
        
        # Logic Gate
        if best_bg['score'] > BG_OCCUPANCY_FLOOR:
            detected = 'empty_bg'
        else:
            is_valid = (best_ore['score'] > ORE_STRICT_GATE) and (margin > MARGIN_REQUIREMENT)
            detected = best_ore['tier'] if is_valid else "low_conf_ore"
            
            # --- TRACK RECOVERY FOR VISUAL AUDIT ---
            # If it passes now but would have failed the old strict settings (0.50/0.10)
            if is_valid and (best_ore['score'] <= 0.50 or margin <= 0.10):
                recovery_detected = True
                color = (0, 255, 255) # Yellow for RECOVERED
            elif is_valid:
                color = (0, 255, 0) # Green for HIGH-CONF
            else:
                color = (0, 0, 255) # Red for MISS

            # Draw labels for the verification images
            rx1, ry1 = int(cx - AI_DIM//2), int(row4_y - AI_DIM//2)
            cv2.rectangle(img_color, (rx1, ry1), (rx1+AI_DIM, ry1+AI_DIM), color, 1)
            cv2.putText(img_color, f"{detected}", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected,
            'ore_score': round(best_ore['score'], 4),
            'bg_score': round(best_bg['score'], 4),
            'margin': round(margin, 4),
            'ore_id': best_ore['id'], 'bg_id': best_bg['id']
        })

    # Only save images where we actually fixed a "Low Confidence" issue
    if recovery_detected:
        out_path = os.path.join(DEBUG_IMG_DIR, f"recovery_f{f_idx}.jpg")
        cv2.imwrite(out_path, img_color)

    return frame_results

def run_ore_audit():
    if not os.path.exists(STEP1_CSV): return
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(1000, len(df)))
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v4.6: RECOVERY MODE ---")
    print(f"Gate: {ORE_STRICT_GATE} | Margin: {MARGIN_REQUIREMENT} | Shadow: {SHADOW_BOOST}")

    all_results = []
    worker_func = partial(process_single_frame, templates=templates, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v4.6_recovery.csv"), index=False)
    
    print(f"\n--- RECOVERY DETECTION SUMMARY (v4.6) ---")
    print(audit_df['detected'].value_counts())
    
    low_conf_count = len(audit_df[audit_df['detected'] == 'low_conf_ore'])
    total_occupied = len(audit_df[audit_df['detected'] != 'empty_bg'])
    print(f"\nRecovery Efficiency: {((total_occupied - low_conf_count) / total_occupied)*100:.1f}% of occupied slots identified.")
    print(f"Check {DEBUG_IMG_DIR} for visual verification of Yellow-labeled ores.")

if __name__ == "__main__":
    run_ore_audit()