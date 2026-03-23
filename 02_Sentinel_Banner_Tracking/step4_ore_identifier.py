# step4_ore_identifier.py
# Purpose: Master Plan Step 4 - Identify all 24 ores on every floor using 
#          the Forensic Trinity (Fusion), Game Rules, and Crosshair Mitigation.
# Version: 1.2 (The Forensic Trinity & Game-Rule Interlock)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

# GRID CONSTANTS (AI SENSOR CENTERS)
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
AI_DIM = 48
SCALE = 1.20
SIDE_PX = int(AI_DIM * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS & GATES
PRISTINE_WINDOW = 15
D_GATE = 6.0          # Occupancy threshold (mean diff vs BG)
MIN_FUSED_GATE = 0.32  # Minimum score to accept a Trinity match

def get_masks():
    """Generates standard circular mask and inner-core mask for crosshairs."""
    std = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(std, (SIDE_PX//2, SIDE_PX//2), int(18 * SCALE), 255, -1)
    core = np.zeros((SIDE_PX, SIDE_PX), dtype=np.uint8)
    cv2.circle(core, (SIDE_PX//2, SIDE_PX//2), int(10 * SCALE), 255, -1)
    return std, core

def get_gradient_map(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_trinity_templates():
    """Loads templates with pre-processed Trinity features (Texture/Silhouette/Grain)."""
    templates = {'active': {}, 'shadow': {}, 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    
    for f in os.listdir(t_path):
        img_raw = cv2.imread(os.path.join(t_path, f), 0)
        if img_raw is None: continue
        img = cv2.resize(img_raw, (SIDE_PX, SIDE_PX))
        
        if f.startswith("background"):
            templates['bg'].append(img)
            continue
            
        if "_plain_" not in f or any(x in f for x in ["player", "negative"]): continue
        
        tier = f.split("_")[0]
        state = 'active' if '_act_' in f else 'shadow'
        
        # Pre-process features
        _, sil = cv2.threshold(cv2.GaussianBlur(img, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gra = get_gradient_map(img)
        
        if tier not in templates[state]: templates[state][tier] = []
        templates[state][tier].append({'img': img, 'sil': sil, 'gra': gra, 'id': f})
        
    return templates

def detect_crosshair(roi_bgr):
    """Detects if a targeting crosshair is present using HSV vibrancy."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Detect Gold/Blue/Red UI elements
    mask = cv2.inRange(hsv, (0, 100, 100), (180, 255, 255))
    return cv2.countNonZero(mask) > 150

def run_ore_identification():
    if not os.path.exists(BOUNDARIES_CSV): return
    df_floors = pd.read_csv(BOUNDARIES_CSV)
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    lib = load_trinity_templates()
    std_mask, core_mask = get_masks()
    
    print(f"--- STEP 4: ORE IDENTIFICATION v1.2 (24-Slot Trinity) ---")
    inventory = []

    for _, floor in df_floors.iterrows():
        f_id, start_f, end_f = int(floor['floor_id']), int(floor['true_start_frame']), int(floor['end_frame'])
        
        # Parse DNA for R3/R4 (Bottom)
        r4_dna, r3_dna = floor['dna_sig'].split('-')
        
        # Determine allowed tiers for this floor (Enforce Game Rules)
        is_boss = f_id in cfg.BOSS_DATA
        if is_boss:
            boss_tier = cfg.BOSS_DATA[f_id].get('tier', 'mixed')
            allowed_tiers = [boss_tier] if boss_tier != 'mixed' else ['dirt1','dirt2','com1','rare1','epic1','leg1']
        else:
            allowed_tiers = [t for t, (s, e) in cfg.ORE_RESTRICTIONS.items() if s <= f_id <= e]

        floor_results = {'floor_id': f_id, 'start_frame': start_f}
        search_limit = min(end_f, start_f + PRISTINE_WINDOW)
        
        print(f"Floor {f_id:03d}: Scanning 24 slots across frames {start_f}-{search_limit}...", end="\r")

        # Scan all 4 Rows (0-3)
        for r_idx in range(4):
            for col in range(6):
                slot_key = f"R{r_idx+1}_S{col}"
                
                # Determine Occupancy
                occupied = False
                if r_idx == 2: occupied = r3_dna[col] == '1' # DNA Row 3
                elif r_idx == 3: occupied = r4_dna[col] == '1' # DNA Row 4
                else:
                    # Top rows: use background subtraction check on start frame
                    img_start = cv2.imread(os.path.join(buffer_dir, all_files[start_f]), 0)
                    y_c, x_c = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                    roi = img_start[y_c-24:y_c+24, x_c-24:x_c+24]
                    if roi.shape == (48,48):
                        diff = min([np.sum(cv2.absdiff(roi, bg_tpl)) / 2304 for bg_tpl in lib['bg']])
                        occupied = diff > D_GATE

                if not occupied:
                    floor_results[slot_key], floor_results[f"{slot_key}_score"] = "empty", 0.0
                    continue

                # Forensic Per-Slot Trinity Search
                best_s, best_t = -1, "low_conf"
                for f_idx in range(start_f, search_limit + 1):
                    img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
                    if img_bgr is None: continue
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    
                    # ROI Extraction
                    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
                    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2
                    roi_bgr = img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                    roi_gray = img_gray[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
                    if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
                    
                    # Crosshair Mitigation
                    active_mask = core_mask if detect_crosshair(roi_bgr) else std_mask
                    
                    # State Selection (Active vs Shadow)
                    state = 'active' if (cv2.Laplacian(roi_gray, cv2.CV_64F).var() > 300 or np.mean(roi_gray) > 90) else 'shadow'
                    
                    # Trinity Matching
                    roi_sil = cv2.threshold(cv2.GaussianBlur(roi_gray, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    roi_gra = get_gradient_map(roi_gray)
                    
                    for tier in allowed_tiers:
                        if tier not in lib[state]: continue
                        for tpl in lib[state][tier]:
                            # Sensor 1: Texture
                            s_tex = cv2.matchTemplate(roi_gray, tpl['img'], cv2.TM_CCOEFF_NORMED, mask=active_mask).max()
                            # Sensor 2: Geometry
                            s_sil = cv2.matchTemplate(roi_sil, tpl['sil'], cv2.TM_CCOEFF_NORMED).max()
                            # Sensor 3: Grain
                            s_gra = cv2.matchTemplate(roi_gra, tpl['gra'], cv2.TM_CCOEFF_NORMED).max()
                            
                            # Fusion (Weighted Mean)
                            fused = (s_tex * 0.4) + (s_sil * 0.3) + (s_gra * 0.3)
                            if fused > best_s:
                                best_s, best_t = fused, tier
                
                floor_results[slot_key] = best_t if best_s > MIN_FUSED_GATE else "low_conf"
                floor_results[f"{slot_key}_score"] = round(float(best_s), 4)

        inventory.append(floor_results)
        
        # PROOF GENERATION
        if f_id % 5 == 0:
            img_audit = cv2.imread(os.path.join(buffer_dir, all_files[start_f]))
            for r_idx in range(4):
                for col in range(6):
                    key = f"R{r_idx+1}_S{col}"
                    if floor_results[key] == "empty": continue
                    ay = int(ORE0_Y + (r_idx * STEP))
                    ax = int(ORE0_X + (col * STEP))
                    color = (0, 255, 0) if floor_results[key] != "low_conf" else (0, 0, 255)
                    cv2.putText(img_audit, floor_results[key], (ax+HUD_DX-25, ay+HUD_DY), 0, 0.4, (0,0,0), 2)
                    cv2.putText(img_audit, floor_results[key], (ax+HUD_DX-25, ay+HUD_DY), 0, 0.4, color, 1)
            cv2.imwrite(os.path.join(VERIFY_DIR, f"audit_f{f_id:03d}.jpg"), img_audit)

    pd.DataFrame(inventory).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Inventory saved. Check {VERIFY_DIR} for results.")

if __name__ == "__main__":
    run_ore_identification()