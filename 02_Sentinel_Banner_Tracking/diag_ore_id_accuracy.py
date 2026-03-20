# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using DNA-Gated Dual-ROI Logic.
# Version: 4.8 (DNA Gate & Multi-Tier Forensic Leaderboard)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "identity_verification")

# FORENSIC TARGETS: Add specific (frame_idx, slot_id) pairs to see a full leaderboard
FORENSIC_TARGETS = [(53, 0), (53, 1), (53, 2), (53, 3), (53, 4), (53, 5)]

# ROI CONSTANTS
DIM_OCC = 30  # Surgical ROI for Background/Occupancy (CCOEFF)
DIM_ID  = 48  # Contextual ROI for Tier Identification (Masked CCORR)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# THRESHOLDS
BG_OCCUPANCY_FLOOR = 0.82  # CCOEFF Background Match
ORE_STRICT_GATE    = 0.80  # CCORR Identity Match (Masked)

def get_spatial_mask():
    """Circular mask for 48x48 Identity phase."""
    mask = np.zeros((DIM_ID, DIM_ID), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    """
    Loads templates in two sizes: 30x30 (Occupancy) and 48x48 (Identity).
    """
    templates = {'ore_occ': {}, 'ore_id': {}, 'bg_occ': [], 'bg_id': []}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        
        # 1. Prepare Occupancy (30x30 Center Crop)
        h, w = img_raw.shape
        cy, cx = h // 2, w // 2
        r = DIM_OCC // 2
        img_occ = img_raw[cy-r : cy+r, cx-r : cx+r]
        
        # 2. Prepare Identity (48x48)
        img_id = cv2.resize(img_raw, (DIM_ID, DIM_ID))
            
        if f.startswith("background") or f.startswith("negative_ui"):
            templates['bg_occ'].append({'id': f, 'img': img_occ})
            templates['bg_id'].append({'id': f, 'img': img_id})
        else:
            parts = f.split("_")
            if len(parts) < 2: continue
            tier, state = parts[0], parts[1]
            
            if tier not in templates['ore_occ']:
                templates['ore_occ'][tier] = {'act': [], 'sha': []}
                templates['ore_id'][tier] = {'act': [], 'sha': []}
            
            if state in ['act', 'sha']:
                templates['ore_occ'][tier][state].append({'id': f, 'img': img_occ})
                templates['ore_id'][tier][state].append({'id': f, 'img': img_id})
                
    return templates

def process_single_frame(frame_data, dna_map, templates, mask, buffer_dir):
    """Worker function using DNA-Gated Dual-ROI logic."""
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return []
    
    # DNA bitmask for this frame
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    frame_results = []
    has_detections = False
    forensic_data = []

    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        
        # --- GATE 1: DNA Occupancy ---
        if r4_dna[col] == '0':
            frame_results.append({
                'frame': f_idx, 'slot': col, 'detected': 'empty_dna',
                'occ_score': 0.0, 'id_score': 0.0, 'ore_id': 'none'
            })
            continue

        # --- GATE 2: OCCUPANCY (30x30 CCOEFF) ---
        x1_occ, y1_occ = int(cx - (DIM_OCC//2) - JITTER), int(row4_y - (DIM_OCC//2) - JITTER)
        search_occ = img_gray[y1_occ : y1_occ + DIM_OCC + (JITTER*2), x1_occ : x1_occ + DIM_OCC + (JITTER*2)]
        
        best_bg_occ = 0.0
        for bg_tpl in templates['bg_occ']:
            res = cv2.matchTemplate(search_occ, bg_tpl['img'], cv2.TM_CCOEFF_NORMED)
            best_bg_occ = max(best_bg_occ, cv2.minMaxLoc(res)[1])

        if best_bg_occ > BG_OCCUPANCY_FLOOR:
            frame_results.append({
                'frame': f_idx, 'slot': col, 'detected': 'empty_bg',
                'occ_score': round(best_bg_occ, 4), 'id_score': 0.0, 'ore_id': 'none'
            })
            continue

        # --- PHASE 2: IDENTITY (48x48 Contextual + Mask) ---
        x1_id, y1_id = int(cx - (DIM_ID//2) - JITTER), int(row4_y - (DIM_ID//2) - JITTER)
        search_id = img_gray[y1_id : y1_id + DIM_ID + (JITTER*2), x1_id : x1_id + DIM_ID + (JITTER*2)]
        
        all_matches = []
        for tier, states in templates['ore_id'].items():
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_id, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    score = cv2.minMaxLoc(res)[1]
                    all_matches.append({'tier': tier, 'id': ore_tpl['id'], 'score': score})
        
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = all_matches[0] if all_matches else {'tier': 'empty', 'score': 0.0, 'id': 'none'}

        # FORENSIC LOGGING: If this slot is a target, capture the leaderboard
        if (f_idx, col) in FORENSIC_TARGETS:
            forensic_data.append({'frame': f_idx, 'slot': col, 'leaderboard': all_matches[:10]})

        is_valid = best_match['score'] > ORE_STRICT_GATE
        detected = best_match['tier'] if is_valid else "low_conf_id"
        
        # Visual Annotation
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        cv2.putText(img_color, f"{detected}", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if is_valid: has_detections = True

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected,
            'occ_score': round(best_bg_occ, 4), 
            'id_score': round(best_match['score'], 4),
            'ore_id': best_match['id']
        })

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"id_f{f_idx}.jpg"), img_color)

    return frame_results, forensic_data

def run_ore_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV):
        print("Error: Missing Step 1 or DNA CSV.")
        return

    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    # 1. Load DNA Gate
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    
    # 2. Setup Processing
    df = pd.read_csv(STEP1_CSV)
    df_sample = df.sample(min(1000, len(df)))
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    print(f"--- ORE ID AUDIT v4.8: DNA-GATED FORENSIC CONSENSUS ---")
    print(f"DNA Gate Active | Processing Occupied Slots Only")

    all_results = []
    all_forensics = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            res, foren = future.result()
            all_results.extend(res)
            all_forensics.extend(foren)
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    # Output Forensics
    if all_forensics:
        print("\n--- FORENSIC LEADERBOARD (Top Targets) ---")
        for f in all_forensics:
            print(f"\nFrame {f['frame']}, Slot {f['slot']}:")
            for rank, entry in enumerate(f['leaderboard']):
                print(f"  #{rank+1} [{entry['score']:.4f}] {entry['id']}")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v4.8_consensus.csv"), index=False)
    
    print(f"\n--- GATED DETECTION SUMMARY (v4.8) ---")
    print(audit_df['detected'].value_counts())
    print(f"\n[DONE] Audit complete. Check forensic leaderboards above for overlap analysis.")

if __name__ == "__main__":
    run_ore_audit()