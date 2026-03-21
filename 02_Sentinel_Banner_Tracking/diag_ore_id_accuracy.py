# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 7.5 (Sobel Gradient Matching & Contrast Normalization)

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

# --- GROUND TRUTH SECTION ---
# Add (frame_idx, slot_id): 'correct_tier' pairs here to analyze failures.
GROUND_TRUTH = {
    (0, 0): 'empty_dna', 
    (0, 1): 'empty_dna',
    (0, 2): 'dirt1',
    (0, 3): 'com1',
    (0, 4): 'com1',
    (0, 5): 'dirt1',
    (1, 0): 'empty_dna', 
    (1, 1): 'empty_dna',
    (1, 2): 'dirt1',
    (1, 3): 'com1',
    (1, 4): 'com1',
    (1, 5): 'dirt1',
    (2, 0): 'empty_dna', 
    (2, 1): 'empty_dna',
    (2, 2): 'dirt1',
    (2, 3): 'com1',
    (2, 4): 'com1',
    (2, 5): 'dirt1',
    (121, 0): 'dirt1', 
    (121, 1): 'dirt1',
    (121, 2): 'empty_dna',
    (121, 3): 'empty_dna',
    (121, 4): 'empty_dna',
    (121, 5): 'dirt1',
    (264, 0): 'empty_dna', 
    (264, 1): 'dirt2',
    (264, 2): 'empty_dna',
    (264, 3): 'epic1',
    (264, 4): 'dirt2',
    (264, 5): 'empty_dna'
}

# ROI CONSTANTS
DIM_ID  = 48  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# THRESHOLDS
ORE_STRICT_GATE = 0.65  # Lowered for Gradient matching (Coefficient is stricter)
Z_SCORE_THRESHOLD = 1.8  
STRUCTURAL_WEIGHT_COEFF = 0.0001 

# BULLY PENALTY MAP (Adjusted for Gradient logic)
BULLY_PENALTIES = {
    'div3_sha_plain_0.png': 0.10,
    'com3_act_pmod_hbar_xhair_0.png': 0.05,
    'leg1_act_pmod_6.png': 0.05
}

# GAME PHYSICS: ORE FLOOR RESTRICTIONS
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_gradient_map(img):
    """Generates a Sobel gradient magnitude map to emphasize structural lines."""
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    mag = np.uint8(np.clip(mag, 0, 255))
    # Apply contrast stretching
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

def get_complexity(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def get_family(tier_name):
    return ''.join([i for i in tier_name if not i.isdigit()])

def get_spatial_mask():
    mask = np.zeros((DIM_ID, DIM_ID), dtype=np.uint8)
    cv2.circle(mask, (24, 24), 18, 255, -1)
    return mask

def load_all_templates():
    templates = {'ore_id': {}}
    t_path = cfg.TEMPLATE_DIR
    if not os.path.exists(t_path): return templates

    for f in os.listdir(t_path):
        if not f.endswith(('.png', '.jpg')): continue
        if f.startswith(("background", "negative_ui")): continue
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        img_id = cv2.resize(img_raw, (DIM_ID, DIM_ID))
        
        # Pre-compute Gradient Map for templates
        grad_map = get_gradient_map(img_id)
        complexity = get_complexity(img_id)
        
        parts = f.split("_")
        if len(parts) < 2: continue
        tier, state = parts[0], parts[1]
        if tier not in templates['ore_id']:
            templates['ore_id'][tier] = []
        templates['ore_id'][tier].append({
            'id': f, 'img': grad_map, 'orig': img_id, 'comp': complexity, 'tier': tier
        })
    return templates

def process_single_frame(frame_data, dna_map, templates, mask, buffer_dir):
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return []
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    slot_matches = {}

    for col in range(6):
        if r4_dna[col] == '0':
            slot_matches[col] = {'status': 'empty_dna', 'candidates': [], 'z_score': 0.0}
            continue

        cx = int(ORE0_X + (col * STEP))
        x1_id, y1_id = int(cx - (DIM_ID//2) - JITTER), int(row4_y - (DIM_ID//2) - JITTER)
        search_id = img_gray[y1_id : y1_id + DIM_ID + (JITTER*2), x1_id : x1_id + DIM_ID + (JITTER*2)]
        if search_id.shape[0] < DIM_ID or search_id.shape[1] < DIM_ID: continue
        
        # Transform ROI to Gradient Space
        grad_roi = get_gradient_map(search_id)
        roi_comp = get_complexity(search_id)
        
        all_candidates = []
        for tier, states in templates['ore_id'].items():
            for tpl in states:
                # Use CCOEFF_NORMED on Gradient maps
                res = cv2.matchTemplate(grad_roi, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=mask)
                score = cv2.minMaxLoc(res)[1]
                
                penalty = BULLY_PENALTIES.get(tpl['id'], 0.0)
                comp_diff = abs(roi_comp - tpl['comp'])
                structural_penalty = comp_diff * STRUCTURAL_WEIGHT_COEFF
                
                weighted_score = score - penalty - structural_penalty
                all_candidates.append({'tier': tier, 'id': tpl['id'], 'score': weighted_score, 'raw': score})
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        scores = [c['score'] for c in all_candidates]
        mean_s = np.mean(scores)
        std_s = np.std(scores) if np.std(scores) > 0 else 1.0
        z_score = (all_candidates[0]['score'] - mean_s) / std_s
        
        slot_matches[col] = {
            'status': 'occupied',
            'candidates': all_candidates, 
            'z_score': z_score,
            'mean_score': mean_s
        }

    # --- ANCHOR ELECTION ---
    anchor = {'tier': 'none', 'score': -1.0, 'z': 0.0, 'range': (1, 999), 'col': -1}
    for col, data in slot_matches.items():
        if data['status'] == 'empty_dna': continue
        if not data['candidates']: continue
        top = data['candidates'][0]
        if data['z_score'] > anchor['z'] and data['z_score'] > Z_SCORE_THRESHOLD:
            anchor = {
                'tier': top['tier'], 'score': top['score'], 'z': data['z_score'],
                'range': ORE_RESTRICTIONS.get(top['tier'], (1, 999)), 'col': col
            }

    # --- RESOLUTION ---
    frame_results = []
    has_detections = False
    family_champions = {}
    
    for col, data in slot_matches.items():
        if data['status'] == 'empty_dna': continue
        for cand in data['candidates']:
            t_range = ORE_RESTRICTIONS.get(cand['tier'], (1, 999))
            if t_range[0] <= anchor['range'][1] and t_range[1] >= anchor['range'][0]:
                fam = get_family(cand['tier'])
                if fam not in family_champions or cand['score'] > family_champions[fam]['score']:
                    family_champions[fam] = {'tier': cand['tier'], 'score': cand['score'], 'id': cand['id']}

    for col in range(6):
        data = slot_matches.get(col)
        if not data: continue
        is_valid = False

        if data['status'] == 'empty_dna':
            detected = 'empty_dna'
            final = {'tier': 'empty_dna', 'score': 0.0, 'id': 'none'}
        else:
            valid_options = []
            for fam, champion in family_champions.items():
                for cand in data['candidates']:
                    if cand['tier'] == champion['tier']:
                        valid_options.append(cand)
                        break
            
            valid_options.sort(key=lambda x: x['score'], reverse=True)
            if not valid_options: 
                detected = 'low_conf_id'
                final = {'tier': 'none', 'score': 0.0, 'id': 'none'}
            else:
                final = valid_options[0]
                is_valid = final['score'] > 0.45 # Gradient matching scores are generally lower
                detected = final['tier'] if is_valid else "low_conf_id"
        
        # --- GROUND TRUTH FORENSICS ---
        truth_tier = GROUND_TRUTH.get((f_idx, col))
        truth_data = {'rank': -1, 'score': 0.0}
        if truth_tier:
            if truth_tier != 'empty_dna':
                for rank, c in enumerate(data.get('candidates', [])):
                    if c['tier'] == truth_tier:
                        truth_data = {'rank': rank + 1, 'score': round(c['score'], 4)}
                        break
        
        # Visual Annotation
        color = (0, 255, 0) if (detected != 'low_conf_id' and detected != 'empty_dna') else (0, 0, 255)
        if detected == 'empty_dna': color = (100, 100, 100)
        if col == anchor['col']: color = (0, 255, 255)
        
        cx = int(ORE0_X + (col * STEP))
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        
        label = f"{detected} Z:{data.get('z_score', 0.0):.1f}"
        if truth_tier and detected != truth_tier:
            cv2.rectangle(img_color, (rx1-2, ry1-2), (rx1+DIM_ID+2, ry1+DIM_ID+2), (255, 0, 0), 1)
            label += f" (T:{truth_data['rank']})"
        
        cv2.putText(img_color, label, (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        if is_valid: has_detections = True

        frame_results.append({
            'frame': f_idx, 'slot': col, 'detected': detected, 
            'score': round(final['score'], 4), 'z_score': round(data.get('z_score', 0.0), 2),
            'truth_tier': truth_tier if truth_tier else 'none',
            'truth_rank': truth_data['rank'], 'truth_score': truth_data['score'],
            'ore_id': final['id'], 'is_anchor': (col == anchor['col'])
        })

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"gradient_v75_f{f_idx}.jpg"), img_color)
    return frame_results

def run_surgical_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    truth_frames = list(set([k[0] for k in GROUND_TRUTH.keys()]))
    df_sample = df[df['frame_idx'].isin(truth_frames)]
    if len(df_sample) < 400:
        remaining = df[~df['frame_idx'].isin(truth_frames)].sample(min(400 - len(df_sample), len(df)))
        df_sample = pd.concat([df_sample, remaining])

    print(f"--- ORE ID AUDIT v7.5: GRADIENT STRUCTURAL MATCHING ---")
    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v7.5_forensic.csv"), index=False)
    
    print(f"\n--- GROUND TRUTH ERROR ANALYSIS ---")
    gt_only = audit_df[audit_df['truth_tier'] != 'none']
    if not gt_only.empty:
        ores_only = gt_only[gt_only['truth_tier'] != 'empty_dna']
        if not ores_only.empty:
            correct_ores = len(ores_only[ores_only['detected'] == ores_only['truth_tier']])
            print(f"Ore Identification Accuracy: {correct_ores}/{len(ores_only)} ({correct_ores/len(ores_only)*100:.1f}%)")
            missed = ores_only[ores_only['detected'] != ores_only['truth_tier']]
            if not missed.empty:
                print(f"  Average Rank of True Ore when missed: {missed['truth_rank'].mean():.1f}")
    print(f"\n--- DETECTION SUMMARY ---")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_surgical_audit()