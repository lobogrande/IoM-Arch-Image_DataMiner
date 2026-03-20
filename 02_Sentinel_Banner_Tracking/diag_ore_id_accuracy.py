# diag_ore_id_accuracy.py
# Purpose: Forensic audit of Row 4 ore identification accuracy using Physical Constraints.
# Version: 5.6 (Probabilistic Floor Profiler & Restriction Gating)

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

# THRESHOLDS
BG_OCCUPANCY_FLOOR = 0.82  
ORE_STRICT_GATE    = 0.75 

# GAME PHYSICS: ORE FLOOR RESTRICTIONS
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_family(tier_name):
    """Extracts family name (e.g., 'dirt1' -> 'dirt')."""
    return ''.join([i for i in tier_name if not i.isdigit()])

def get_spatial_mask():
    """Circular mask for 48x48 Identity phase."""
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
        
        # Occupancy (30x30)
        h, w = img_raw.shape
        cy, cx = h // 2, w // 2
        r = DIM_OCC // 2
        img_occ = img_raw[cy-r : cy+r, cx-r : cx+r]
        # Identity (48x48)
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
    f_idx = frame_data['frame_idx']
    filename = frame_data['filename']
    img_path = os.path.join(buffer_dir, filename)
    img_color = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) if img_color is not None else None
    if img_gray is None: return []
    
    r4_dna = dna_map.get(f_idx, "000000")
    row4_y = int(ORE0_Y + (3 * STEP))
    slot_matches = {}

    # --- PASS 1: WIDE IDENTIFICATION (Find the Anchor) ---
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
        
        tier_perf = {} 
        for tier, states in templates['ore_id'].items():
            best_s = -1.0
            best_id = 'none'
            for state in ['act', 'sha']:
                for ore_tpl in states[state]:
                    res = cv2.matchTemplate(search_id, ore_tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                    score = cv2.minMaxLoc(res)[1]
                    if score > best_s:
                        best_s = score
                        best_id = ore_tpl['id']
            tier_perf[tier] = {'score': best_s, 'id': best_id}
            
        slot_matches[col] = {'status': 'occupied', 'occ_score': best_bg_occ, 'tiers': tier_perf}

    # --- PASS 2: FLOOR RESTRICTION ELECTION ---
    # Find the "Anchor" - the identification with the highest confidence in the whole row.
    anchor = {'tier': 'none', 'score': 0.0, 'range': (1, 999)}
    for col, data in slot_matches.items():
        if data['status'] != 'occupied': continue
        for tier, perf in data['tiers'].items():
            if perf['score'] > anchor['score']:
                anchor = {'tier': tier, 'score': perf['score'], 'range': ORE_RESTRICTIONS.get(tier, (1, 999))}

    # --- PASS 3: LOGICAL CONSENSUS (Restrict everything to the Anchor's range) ---
    frame_results = []
    has_detections = False
    
    # Also find Champion Tiers within the valid range (One Tier Per Family)
    family_champions = {}
    for col, data in slot_matches.items():
        if data['status'] != 'occupied': continue
        for tier, perf in data['tiers'].items():
            t_range = ORE_RESTRICTIONS.get(tier, (1, 999))
            # Must overlap with Anchor range
            if t_range[0] <= anchor['range'][1] and t_range[1] >= anchor['range'][0]:
                fam = get_family(tier)
                if fam not in family_champions or perf['score'] > family_champions[fam]['score']:
                    family_champions[fam] = {'tier': tier, 'score': perf['score']}

    for col in range(6):
        data = slot_matches[col]
        if data['status'] == 'empty_dna':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_dna'})
            continue
        if data['status'] == 'empty_bg':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_bg'})
            continue

        # Filter candidates based on: 
        # 1. Champion Tier for family 
        # 2. Within Anchor Floor Range
        candidates = []
        for fam, champion in family_champions.items():
            t_range = ORE_RESTRICTIONS.get(champion['tier'], (1, 999))
            if t_range[0] <= anchor['range'][1] and t_range[1] >= anchor['range'][0]:
                perf = data['tiers'].get(champion['tier'])
                if perf:
                    candidates.append({'tier': champion['tier'], 'score': perf['score'], 'id': perf['id']})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        if not candidates:
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'low_conf_id'})
            continue
        
        final = candidates[0]
        is_valid = final['score'] > ORE_STRICT_GATE
        detected = final['tier'] if is_valid else "low_conf_id"
        
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cx = int(ORE0_X + (col * STEP))
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        cv2.putText(img_color, f"{detected} ({final['score']:.2f})", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if is_valid: has_detections = True

        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final['score'], 4), 'ore_id': final['id']})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"restricted_v56_f{f_idx}.jpg"), img_color)

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
    
    df_sample = df.sample(min(1000, len(df)))
    print(f"--- ORE ID AUDIT v5.6: PROBABILISTIC FLOOR PROFILER ---")
    print(f"Restriction Map: {len(ORE_RESTRICTIONS)} tiers | Anchor-Gated Consistency")

    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v5.6_restricted.csv"), index=False)
    print(f"\n--- RESTRICTED CONSENSUS SUMMARY ---")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_ore_audit()