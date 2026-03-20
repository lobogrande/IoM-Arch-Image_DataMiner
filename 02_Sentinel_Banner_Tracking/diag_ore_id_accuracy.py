# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using DNA-Aligned ROI.
# Version: 4.5 (Waterfall Sensitivity & Forensic Leaderboards)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "debug_visuals")

# ROI CONSTANTS
AI_DIM = 30  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# CURRENT PRODUCTION THRESHOLDS
BG_OCCUPANCY_FLOOR = 0.82  
ORE_STRICT_GATE = 0.50     
MARGIN_REQUIREMENT = 0.10  

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
    
    img_gray = cv2.imread(img_path, 0)
    if img_gray is None: return []
    
    row4_y = int(ORE0_Y + (3 * STEP))
    frame_results = []

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        x1, y1 = int(cx - (AI_DIM//2) - JITTER), int(row4_y - (AI_DIM//2) - JITTER)
        search_area = img_gray[y1 : y1 + AI_DIM + (JITTER*2), x1 : x1 + AI_DIM + (JITTER*2)]
        if search_area.shape[0] < AI_DIM or search_area.shape[1] < AI_DIM: continue

        # 1. Background Leaderboard
        bg_matches = []
        for bg_tpl in templates['bg']:
            res = cv2.matchTemplate(search_area, bg_tpl['img'], cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            bg_matches.append({'id': bg_tpl['id'], 'score': val})
        bg_matches.sort(key=lambda x: x['score'], reverse=True)
        best_bg = bg_matches[0]

        # 2. Ore Leaderboard
        ore_matches = []
        for tier, states in templates['ore'].items():
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_area, ore_tpl['img'], cv2.TM_CCOEFF_NORMED)
                    _, score, _, _ = cv2.minMaxLoc(res)
                    if state == 'sha': score *= 1.02
                    ore_matches.append({'id': ore_tpl['id'], 'tier': tier, 'score': score})
        ore_matches.sort(key=lambda x: x['score'], reverse=True)
        best_ore = ore_matches[0]

        margin = best_ore['score'] - best_bg['score']
        
        # Determine status based on production thresholds
        if best_bg['score'] > BG_OCCUPANCY_FLOOR:
            detected = 'empty_bg'
        else:
            is_valid = (best_ore['score'] > ORE_STRICT_GATE) and (margin > MARGIN_REQUIREMENT)
            detected = best_ore['tier'] if is_valid else "low_conf_ore"

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected,
            'ore_score': round(best_ore['score'], 4),
            'bg_score': round(best_bg['score'], 4),
            'margin': round(margin, 4),
            'ore_id': best_ore['id'], 'bg_id': best_bg['id'],
            'top_ores': ore_matches[:3], # For forensics
            'top_bgs': bg_matches[:3]    # For forensics
        })

    return frame_results

def run_ore_audit():
    if not os.path.exists(STEP1_CSV): return
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(1000, len(df)))
    templates = load_all_templates()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v4.5: FORENSIC DIAGNOSTICS ---")

    all_results = []
    worker_func = partial(process_single_frame, templates=templates, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    
    # --- DIAGNOSTIC 1: SENSITIVITY WATERFALL ---
    print("\n[DIAGNOSTIC] Waterfall Sensitivity Analysis:")
    print(f"{'Gate':<6} | {'Margin':<8} | {'Detections':<12}")
    print("-" * 35)
    for gate in [0.40, 0.45, 0.50, 0.55]:
        for marg in [0.06, 0.08, 0.10, 0.12]:
            # Count how many would be detected at this setting
            # We filter out where bg_score > occupancy floor
            potential = audit_df[audit_df['bg_score'] <= BG_OCCUPANCY_FLOOR]
            count = len(potential[(potential['ore_score'] > gate) & (potential['margin'] > marg)])
            print(f"{gate:<6.2f} | {marg:<8.2f} | {count:<12}")

    # --- DIAGNOSTIC 2: FORENSIC LEADERBOARD ---
    print("\n[DIAGNOSTIC] Forensic Leaderboard (Top 5 'Almost' Ores):")
    # Find low_conf_ore entries with the highest margins
    low_conf = audit_df[audit_df['detected'] == 'low_conf_ore'].sort_values('margin', ascending=False).head(5)
    
    for i, (_, row) in enumerate(low_conf.iterrows()):
        print(f"\nCase #{i+1}: Frame {row['frame']}, Slot {row['slot']} (Margin: {row['margin']:.4f})")
        print(f"  ORE LEADERBOARD: ", end="")
        print(" | ".join([f"{m['id']} ({m['score']:.3f})" for m in row['top_ores']]))
        print(f"  BG LEADERBOARD:  ", end="")
        print(" | ".join([f"{m['id']} ({m['score']:.3f})" for m in row['top_bgs']]))

    audit_df.drop(columns=['top_ores', 'top_bgs']).to_csv(os.path.join(OUT_DIR, "ore_id_v4.5_forensic.csv"), index=False)
    print(f"\n[DONE] Detection Summary (Current Settings):")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_ore_audit()