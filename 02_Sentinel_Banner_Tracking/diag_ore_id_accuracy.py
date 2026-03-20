# diag_ore_id_accuracy.py
# Purpose: Forensic Ore Identification with Structural and Physical Constraints.
# Version: 7.0 (The Discriminator: Greediness Penalties & Structural Gating)

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
DIM_ID  = 48  
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
JITTER = 2 

# THRESHOLDS
ORE_STRICT_GATE = 0.78  # High gate for the initial "Anchor" identification
TIER_CONF_BUFFER = 0.04 # v5.0 logic: prefer lower tiers if scores are close

# BULLY PENALTY MAP (Derived from v6.2 Greediness Index)
# These templates were found to be "Noise Magnets" and receive a scoring penalty.
BULLY_PENALTIES = {
    'div3_sha_plain_0.png': 0.12,
    'com3_act_pmod_hbar_xhair_0.png': 0.08,
    'rare1_act_pmod_6.png': 0.05,
    'rare1_act_plain_3.png': 0.05,
    'dirt1_act_pmod_9.png': 0.04,
    'leg2_act_xhair_0.png': 0.04,
    'dirt1_act_pmod_2.png': 0.04,
    'div2_sha_pmod_1.png': 0.06,
    'dirt2_act_xhair_0.png': 0.04,
    'dirt1_act_plain_2.png': 0.04
}

# GAME PHYSICS: ORE FLOOR RESTRICTIONS
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

def get_complexity(img):
    """Measures visual detail density (Laplacian variance)."""
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
        complexity = get_complexity(img_id)
        
        parts = f.split("_")
        if len(parts) < 2: continue
        tier, state = parts[0], parts[1]
        
        if tier not in templates['ore_id']:
            templates['ore_id'][tier] = []
        templates['ore_id'][tier].append({
            'id': f, 'img': img_id, 'comp': complexity, 'tier': tier
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

    # --- PASS 1: WIDE DISCRIMINATION (Find the Anchor) ---
    for col in range(6):
        if r4_dna[col] == '0': continue 

        cx = int(ORE0_X + (col * STEP))
        x1_id, y1_id = int(cx - (DIM_ID//2) - JITTER), int(row4_y - (DIM_ID//2) - JITTER)
        search_id = img_gray[y1_id : y1_id + DIM_ID + (JITTER*2), x1_id : x1_id + DIM_ID + (JITTER*2)]
        
        if search_id.shape[0] < DIM_ID or search_id.shape[1] < DIM_ID: continue
        roi_comp = get_complexity(search_id)

        all_candidates = []
        for tier, states in templates['ore_id'].items():
            for tpl in states:
                res = cv2.matchTemplate(search_id, tpl['img'], cv2.TM_CCORR_NORMED, mask=mask)
                score = cv2.minMaxLoc(res)[1]
                
                # Apply Greediness Penalty
                penalty = BULLY_PENALTIES.get(tpl['id'], 0.0)
                
                # Apply Structural Gating (Complexity Mismatch)
                # Penalize "Smooth" templates on "Busy" ROIs and vice versa
                comp_diff = abs(roi_comp - tpl['comp'])
                structural_penalty = comp_diff * 0.0001
                
                final_score = score - penalty - structural_penalty
                all_candidates.append({
                    'tier': tier, 'id': tpl['id'], 'score': final_score, 'raw': score
                })
        
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        slot_matches[col] = {'candidates': all_candidates, 'roi_comp': roi_comp}

    if not slot_matches: return []

    # --- PASS 2: ANCHOR-GATED FLOOR PROFILE ---
    # Find the single most mathematically certain ore in the row.
    anchor = {'tier': 'none', 'score': 0.0, 'range': (1, 999)}
    for col, data in slot_matches.items():
        top = data['candidates'][0]
        if top['score'] > anchor['score']:
            anchor = {'tier': top['tier'], 'score': top['score'], 'range': ORE_RESTRICTIONS.get(top['tier'], (1, 999))}

    # --- PASS 3: RESTRICTED RESOLUTION ---
    frame_results = []
    has_detections = False
    
    # Vote on Champion Tiers within the Valid Floor Range
    family_champions = {}
    for col, data in slot_matches.items():
        for cand in data['candidates']:
            # Only allow ores that overlap with the Anchor's floor range
            t_range = ORE_RESTRICTIONS.get(cand['tier'], (1, 999))
            if t_range[0] <= anchor['range'][1] and t_range[1] >= anchor['range'][0]:
                fam = get_family(cand['tier'])
                if fam not in family_champions or cand['score'] > family_champions[fam]['score']:
                    family_champions[fam] = {'tier': cand['tier'], 'score': cand['score'], 'id': cand['id']}

    for col in range(6):
        if r4_dna[col] == '0':
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'empty_dna'})
            continue
            
        data = slot_matches.get(col)
        if not data: continue

        # Filter slot to only allowed tiers from the Row-Champion list
        valid_options = []
        for fam, champion in family_champions.items():
            # Find the local score for this champion tier
            for cand in data['candidates']:
                if cand['tier'] == champion['tier']:
                    valid_options.append(cand)
                    break
        
        valid_options.sort(key=lambda x: x['score'], reverse=True)
        if not valid_options: 
            frame_results.append({'frame': f_idx, 'slot': col, 'detected': 'low_conf_id'})
            continue

        # Final Choice with v5.0 Tier Bias (prefer simpler ores if scores are close)
        final = valid_options[0]
        for challenger in valid_options[1:]:
            if challenger['score'] > (final['score'] - TIER_CONF_BUFFER):
                if 'dirt' in final['tier'] and 'dirt' not in challenger['tier']:
                    pass # Keep dirt as simple
                elif TIER_RANK_VAL(challenger['tier']) < TIER_RANK_VAL(final['tier']):
                    final = challenger

        is_valid = final['score'] > 0.65 # Lowered because penalties reduce absolute scores
        detected = final['tier'] if is_valid else "low_conf_id"
        
        # Visual Verification
        color = (0, 255, 0) if is_valid else (0, 0, 255)
        cx = int(ORE0_X + (col * STEP))
        rx1, ry1 = int(cx - DIM_ID//2), int(row4_y - DIM_ID//2)
        cv2.rectangle(img_color, (rx1, ry1), (rx1+DIM_ID, ry1+DIM_ID), color, 1)
        cv2.putText(img_color, f"{detected} ({final['score']:.2f})", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if is_valid: has_detections = True

        frame_results.append({'frame': f_idx, 'slot': col, 'detected': detected, 'score': round(final['score'], 4), 'ore_id': final['id']})

    if has_detections:
        cv2.imwrite(os.path.join(DEBUG_IMG_DIR, f"discriminator_v70_f{f_idx}.jpg"), img_color)

    return frame_results

def TIER_RANK_VAL(tier):
    """Simple rarity rank for tie-breaking."""
    ranks = {'dirt': 1, 'com': 2, 'rare': 3, 'epic': 4, 'myth': 5, 'leg': 6, 'div': 7}
    return ranks.get(get_family(tier), 0)

def run_discriminator_audit():
    if not os.path.exists(STEP1_CSV) or not os.path.exists(DNA_CSV): return
    if not os.path.exists(DEBUG_IMG_DIR): os.makedirs(DEBUG_IMG_DIR)
    
    dna_df = pd.read_csv(DNA_CSV, dtype={'r4_dna': str})
    dna_map = dna_df.set_index('frame_idx')['r4_dna'].to_dict()
    df = pd.read_csv(STEP1_CSV)
    
    templates = load_all_templates()
    mask = get_spatial_mask()
    buffer_dir = cfg.get_buffer_path(0)
    
    df_sample = df.sample(min(400, len(df)))
    print(f"--- ORE ID AUDIT v7.0: THE DISCRIMINATOR ---")
    print(f"Gating: Laplacian Structural Mismatch + Greediness Penalties")

    all_results = []
    worker_func = partial(process_single_frame, dna_map=dna_map, templates=templates, mask=mask, buffer_dir=buffer_dir)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = df_sample.to_dict('records')
        futures = {executor.submit(worker_func, task): task for task in tasks}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            all_results.extend(future.result())
            if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(df_sample)} frames...")

    audit_df = pd.DataFrame(all_results)
    audit_df.to_csv(os.path.join(OUT_DIR, "ore_id_v7.0_discriminator.csv"), index=False)
    print(f"\n--- DISCRIMINATOR SUMMARY ---")
    print(audit_df['detected'].value_counts())

if __name__ == "__main__":
    run_discriminator_audit()