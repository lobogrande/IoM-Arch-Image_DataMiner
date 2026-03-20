# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using Adaptive Feature Profiling.
# Version: 5.4 (Adaptive Signature Matching & Auto-Tuning Election)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# CONFIG
STEP1_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "sprite_homing_run_0.csv")
DNA_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "dna_sensor_final.csv")
OUT_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_id_audit")
DEBUG_IMG_DIR = os.path.join(OUT_DIR, "identity_verification")

# ROI CONSTANTS
DIM_OCC = 30  
DIM_ID  = 48  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# THRESHOLDS (Starting points, tuned by Auto-Tuner)
BG_OCCUPANCY_FLOOR = 0.82  
ORE_STRICT_GATE    = 0.78  
TIER_CONF_BUFFER   = 0.04  

def get_family(tier_name):
    return ''.join([i for i in tier_name if not i.isdigit()])

def get_feature_signature(img):
    """Calculates a multi-dimensional visual signature for an image."""
    mean = np.mean(img)
    var = cv2.Laplacian(img, cv2.CV_64F).var()
    # Edge Density (Sobel)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edge_density = np.mean(np.sqrt(sobelx**2 + sobely**2))
    return {'mean': mean, 'var': var, 'edges': edge_density}

def get_spatial_mask():
    mask = np.zeros((DIM_ID, DIM_ID), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    templates = {'ore_occ': {}, 'ore_id': {}, 'bg_occ': [], 'bg_id': []}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        
        # 1. Occupancy (30x30)
        h, w = img_raw.shape
        cy, cx = h // 2, w // 2
        r = DIM_OCC // 2
        img_occ = img_raw[cy-r : cy+r, cx-r : cx+r]
        
        # 2. Identity (48x48)
        img_id = cv2.resize(img_raw, (DIM_ID, DIM_ID))
        sig = get_feature_signature(img_id)
            
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
                templates['ore_occ'][tier][state].append({'id': f, 'img': img_occ, 'sig': sig})
                templates['ore_id'][tier][state].append({'id': f, 'img': img_id, 'sig': sig})
    return templates

def process_single_frame(frame_data, dna_map, templates, mask, buffer_dir, penalty_coeff):
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return [], []
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    slot_matches = {}

    # PASS 1: Signature-Based Matching
    for col in range(6):
        cx = int(ORE0_X + (col * STEP))
        if r4_dna[col] == '0':
            slot_matches[col] = {'status': 'empty_dna'}
            continue

        x1_occ, y1_occ = int(cx - (DIM_OCC//2) - JITTER), int(row4_y - (DIM_OCC//2) - JITTER)
        search_occ = img_gray[y1_occ : y1_occ + DIM_OCC + (JITTER*2), x1_occ : x1_occ + DIM_OCC + (JITTER*2)]
        best_bg_occ = 0.0
        for bg_tpl in templates['bg_occ']:
            res = cv2.matchTemplate(search_occ, bg_tpl['img'], cv2.TM_CCOEFF_NORMED)
            best_bg_occ = max(best_bg_occ, cv2.minMaxLoc(res)[1])

        if best_bg_occ > BG_OCCUPANCY_FLOOR:
            slot_matches[col] = {'status': 'empty_bg', 'occ_score': best_bg_occ}
            continue

        x1_id, y1_id = int(cx - (DIM_ID//2) - JITTER), int(row4_y - (DIM_ID//2) - JITTER)
        search_id = img_gray[y1_id : y1_id + DIM_ID + (JITTER*2), x1_id : x1_id + DIM_ID + (JITTER*2)]
        roi_sig = get_feature_signature(search_id)
        
        tier_perf = {} 
        for tier, states in templates['ore_id'].items():
            best_s = -1.0
            best_id = 'none'
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_id, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    raw_score = cv2.minMaxLoc(res)[1]
                    
                    # Distance-Based Penalty: Reject based on visual "weight" mismatch
                    # If ROI is dense/complex and template is smooth, penalize heavily
                    dist = abs(roi_sig['var'] - ore_tpl['sig']['var']) + abs(roi_sig['edges'] - ore_tpl['sig']['edges'])*10
                    penalty = dist * penalty_coeff
                    
                    score = raw_score - penalty
                    if score > best_s:
                        best_s = score
                        best_id = ore_tpl['id']
            tier_perf[tier] = {'score': best_s, 'id': best_id}
            
        slot_matches[col] = {'status': 'occupied', 'occ_score': best_bg_occ, 'tiers': tier_perf}

    # PASS 2: HARD ROW-LEVEL ELECTION
    # Establish the Absolute Champion Profile for this frame
    frame_profile = {} # family -> tier
    for fam in set(get_family(t) for t in templates['ore_id'].keys()):
        best_score = -1.0
        best_tier = 'none'
        for col, data in slot_matches.items():
            if data['status'] != 'occupied': continue
            for tier, perf in data['tiers'].items():
                if get_family(tier) == fam and perf['score'] > best_score:
                    best_score = perf['score']
                    best_tier = tier
        if best_tier != 'none': frame_profile[fam] = best_tier

    # PASS 3: RESOLUTION
    frame_results = []
    has_detections = False
    for col in range(6):
        data = slot_matches[col]
        if data['status'] == 'empty_dna':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_dna'})
            continue
        if data['status'] == 'empty_bg':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_bg'})
            continue

        # Force only the allowed tiers from the profile
        candidates = []
        for fam, champion in frame_profile.items():
            perf = data['tiers'].get(champion)
            if perf: candidates.append({'tier': champion, 'score': perf['score'], 'id': perf['id']})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        if not candidates: continue
        
        final = candidates[0]
        # Tier-Biased Conservation check
        for challenger in candidates[1:]:
            if challenger['score'] > (final['score'] - TIER_CONF_BUFFER):
                # Prefer Common over Dirt, Rare over Common, etc. if scores are close
                if 'dirt' in final['tier'] and 'dirt' not in challenger['tier']:
                    final = challenger

        is_valid = final['score'] > ORE_STRICT_GATE
        detected = final['tier'] if is_valid else "low_conf_id"
        
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cx = int(ORE0_X + (col * STEP))
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        cv2.putText(img_color, f"{detected}", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if is_valid: has_detections = True

        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final['score'], 4), 'ore_id': final['id']})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"adaptive_v54_f{f_idx}.jpg"), img_color)

    return frame_results

def run_ore_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    # --- AUTO-TUNING PRE-FLIGHT ---
    print("--- AUTO-TUNING PARAMETERS ---")
    tuning_sample = df.sample(min(50, len(df)))
    best_penalty = 0.0001
    max_diversity = 0
    
    for test_penalty in [0.00005, 0.0001, 0.0002, 0.0004]:
        results = []
        for _, row in tuning_sample.iterrows():
            res = process_single_frame(row, dna_map, templates, mask, buffer_dir, test_penalty)
            results.extend(res)
        
        # Count non-dirt detections
        diversity = sum(1 for r in results if 'detected' in r and r['detected'] not in ['empty_dna', 'empty_bg', 'low_conf_id'] and 'dirt' not in r['detected'])
        if diversity > max_diversity:
            max_diversity = diversity
            best_penalty = test_penalty

    print(f"Optimal Structural Penalty selected: {best_penalty:.5f}")
    
    # --- MAIN AUDIT ---
    df_sample = df.sample(min(1000, len(df)))
    print(f"--- ORE ID AUDIT v5.4: ADAPTIVE FEATURE PROFILING ---")

    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir, penalty_coeff=best_penalty)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v5.4_adaptive.csv"), index=False)
    print(f"\n--- ADAPTIVE DETECTION SUMMARY ---")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_ore_audit()