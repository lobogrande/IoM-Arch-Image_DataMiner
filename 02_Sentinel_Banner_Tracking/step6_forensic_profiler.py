# step6_audit_profiler.py
# Purpose: Diagnostic Tool - Extract raw sensor metrics (Complexity, Texture, Geometry, Grain)
#          to establish data-driven baselines for tier identification.
# Version: 2.0 (Architecture Aligned & Validated Constants)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- DYNAMIC CONFIGURATION ---
SOURCE_DIR = cfg.get_buffer_path()
RUN_ID = os.path.basename(SOURCE_DIR).split('_')[-1]

# INPUT/OUTPUT PATHS
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"final_floor_boundaries_run_{RUN_ID}.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"floor_dna_inventory_run_{RUN_ID}.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], f"trinity_sensor_profile_run_{RUN_ID}.csv")

# --- VALIDATED GRID CONSTANTS ---
ORE0_X, ORE0_Y = 74, 261
STEP = 59.0
SIDE_PX = 48

# Diagnostic Settings
LIMIT_FLOORS = 20

def get_complexity(img):
    """Calculates Laplacian variance as a measure of structural texture."""
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def apply_clahe(img):
    """Normalizes contrast to expose structural grain."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def get_silhouette(img_gray):
    """Produces binary geometry map for shape matching."""
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure ore (center) is white
    if thresh[SIDE_PX//2, SIDE_PX//2] == 0: thresh = cv2.bitwise_not(thresh)
    return thresh

def get_gradient_map(img_gray):
    """Produces Sobel-based edge magnitude map for grain matching."""
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_templates():
    """Loads ore standards and pre-calculates their Trinity blueprints."""
    res = {}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        if "_act_plain_" in f and not any(x in f for x in["player", "background"]):
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is None: continue
            img_scaled = cv2.resize(img, (SIDE_PX, SIDE_PX))
            tier = f.split("_")[0]
            if tier not in res: res[tier] = []
            img_tex = apply_clahe(img_scaled)
            res[tier].append({
                'tex': img_tex, 'geo': get_silhouette(img_tex), 'gra': get_gradient_map(img_tex),
                'comp': get_complexity(img_scaled)
            })
    return res

def profile_floor(floor_data, dna_map, buffer_dir, all_files, templates):
    f_id = int(floor_data['floor_id'])
    start_f = int(floor_data['true_start_frame'])
    dna_row = dna_map[dna_map['floor_id'] == f_id].iloc[0]
    
    # Analyze the initial "Pristine" state of the floor
    img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[start_f]))
    if img_bgr is None: return[]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    floor_profiles =[]
    is_boss = f_id in cfg.BOSS_DATA if hasattr(cfg, 'BOSS_DATA') else False
    
    for r_idx in range(4):
        for col in range(6):
            key = f"R{r_idx+1}_S{col}"
            if str(dna_row[key]) == '0': continue
            
            cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col * STEP))
            y1, x1 = int(cy - SIDE_PX//2), int(cx - SIDE_PX//2)
            roi_gray = img_gray[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
            
            # 1. Structural Energy (Affinity)
            comp = get_complexity(roi_gray)
            
            # 2. Side-Slice Variance (Matches Step 6 forensic sliver logic)
            slice_roi = roi_gray[15:40, 1:3]
            slice_std = np.std(slice_roi)
            
            # 3. Trinity Pre-processing
            roi_tex = apply_clahe(roi_gray)
            roi_geo = get_silhouette(roi_tex)
            roi_gra = get_gradient_map(roi_tex)
            
            # Center 30x30 crop to avoid border noise
            c_tex, c_geo, c_gra = roi_tex[9:39, 9:39], roi_geo[9:39, 9:39], roi_gra[9:39, 9:39]
            
            def get_raw_trinity(tier):
                if tier not in templates: return 0, 0, 0
                t = templates[tier][0]
                t_tex, t_geo, t_gra = t['tex'][9:39, 9:39], t['geo'][9:39, 9:39], t['gra'][9:39, 9:39]
                s_tex = cv2.minMaxLoc(cv2.matchTemplate(t_tex, c_tex, cv2.TM_CCOEFF_NORMED))[1]
                s_geo = cv2.minMaxLoc(cv2.matchTemplate(t_geo, c_geo, cv2.TM_CCOEFF_NORMED))[1]
                s_gra = cv2.minMaxLoc(cv2.matchTemplate(t_gra, c_gra, cv2.TM_CCOEFF_NORMED))[1]
                return s_tex, s_geo, s_gra

            # Sample extremes (Dirt1 vs Rare1) to find separation gaps
            s_dirt_tex, s_dirt_geo, s_dirt_gra = get_raw_trinity('dirt1')
            s_rare_tex, s_rare_geo, s_rare_gra = get_raw_trinity('rare1')

            profile = {
                'floor': f_id, 'slot': key, 'is_boss': is_boss,
                'roi_complexity': round(comp, 2),
                'side_slice_std': round(slice_std, 2),
                'raw_dirt1_tex': round(s_dirt_tex, 4), 'raw_dirt1_geo': round(s_dirt_geo, 4), 'raw_dirt1_gra': round(s_dirt_gra, 4),
                'raw_rare1_tex': round(s_rare_tex, 4), 'raw_rare1_geo': round(s_rare_geo, 4), 'raw_rare1_gra': round(s_rare_gra, 4)
            }
            floor_profiles.append(profile)
            
    return floor_profiles

def run_profiler():
    print(f"--- STEP 6 AUDIT: TRINITY SENSOR PROFILER (Run {RUN_ID}) ---")
    
    if not os.path.exists(BOUNDARIES_CSV) or not os.path.exists(DNA_INVENTORY_CSV):
        print("Error: Input CSVs missing. Run Steps 3 and 5 first.")
        return

    df_floors = pd.read_csv(BOUNDARIES_CSV)
    if LIMIT_FLOORS: df_floors = df_floors.head(LIMIT_FLOORS)
    
    df_dna = pd.read_csv(DNA_INVENTORY_CSV)
    all_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.endswith(('.png', '.jpg'))])
    templates = load_templates()
    
    all_data =[]
    worker = partial(profile_floor, dna_map=df_dna, buffer_dir=SOURCE_DIR, all_files=all_files, templates=templates)
    
    print(f"Profiling {len(df_floors)} floors...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker, row): row['floor_id'] for _, row in df_floors.iterrows()}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            count += 1
            all_data.extend(future.result())
            print(f"  Processed Floor {count}/{len(df_floors)}...", end="\r")

    pd.DataFrame(all_data).to_csv(OUT_CSV, index=False)
    print(f"\n[DONE] Baseline metrics saved to: {os.path.basename(OUT_CSV)}")

if __name__ == "__main__":
    run_profiler()