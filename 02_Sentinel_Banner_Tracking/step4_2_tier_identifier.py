# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using the Forensic Trinity:
#          Triple-Sensor Fusion, Structural Disparity Guard, and Player-First Forensics.
# Version: 3.6 (The Probabilistic Physical Anchor)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter, defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- 1. GRID & HUD CONSTANTS (Verified) ---
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

# DIAGNOSTIC CONTROL
LIMIT_FLOORS = 20  # Set to None for production

# --- 2. DATA-DRIVEN CONSTANTS (Forensically Calibrated) ---
SIDE_SLICE_WIDTH = 12        
SIDE_SLICE_STD_MAX = 13.0    # DATA-DRIVEN: Captures Floor 3/7 background jitter
Z_TRUST_THRESHOLD = 1.5      
MIN_MOMENTUM_GATE = 4.0      
PLAYER_PRESENCE_TRESHOLD = 0.45 
COMPLEXITY_DIRT_CEILING = 480.0
COMPLEXITY_HIGH_FLOOR = 750.0

# Trinity Weights: Balanced for robust signal extraction
W_TEX, W_GEO, W_GRA = 0.40, 0.30, 0.30
HARVEST_COUNT = 15          

def get_complexity(img):
    if img is None or img.size == 0: return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def get_silhouette(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh[SIDE_PX//2, SIDE_PX//2] == 0: thresh = cv2.bitwise_not(thresh)
    return thresh

def get_gradient_map(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def load_resources():
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    print("Blueprinting Trinity Template Library...")
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_scaled = cv2.resize(img, (SIDE_PX, SIDE_PX))
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = {'tpls': [], 'mean_comp': 0.0}
            img_tex = apply_clahe(img_scaled)
            res['ores'][tier]['tpls'].append({
                'tex': img_tex, 'geo': get_silhouette(img_tex), 'gra': get_gradient_map(img_tex),
                'comp': get_complexity(img_scaled)
            })
        if "negative_player" in f: res['player'].append(img_scaled)
        if "background_plain" in f: res['bg'].append(img_scaled)
    for tier in res['ores']:
        res['ores'][tier]['mean_comp'] = np.mean([t['comp'] for t in res['ores'][tier]['tpls']])
    return res

def check_side_slice_empty(roi_gray):
    """Forensic Gatekeeper: Uses the 13.0 StdDev boundary to find hidden background."""
    slot_48 = roi_gray[4:52, 4:52]
    slice_roi = slot_48[:, 0:SIDE_SLICE_WIDTH]
    std_val = np.std(slice_roi)
    return std_val, std_val < SIDE_SLICE_STD_MAX

def identify_consensus(f_range, r_idx, col_idx, buffer_dir, all_files, allowed_tiers, res):
    """Consensus engine with Player-First forensic priority."""
    frame_candidates = []
    cy, cx = int(ORE0_Y + (r_idx * STEP)), int(ORE0_X + (col_idx * STEP))
    y1, x1 = cy - SIDE_PX//2, cx - SIDE_PX//2 
    is_banner = (r_idx == 0 and col_idx in [2, 3])
    
    peak_p_score, best_roi_gray = 0.0, None

    # 1. Observation Phase
    for f_idx in f_range:
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]))
        if img_bgr is None: continue
        roi_gray = cv2.cvtColor(img_bgr[y1:y1+SIDE_PX, x1:x1+SIDE_PX], cv2.COLOR_BGR2GRAY)
        if roi_gray.shape != (SIDE_PX, SIDE_PX): continue
        
        roi_30 = roi_gray[13:43, 13:43]
        max_p = max([cv2.minMaxLoc(cv2.matchTemplate(pt, roi_30, cv2.TM_CCOEFF_NORMED))[1] for pt in res['player']] + [0])
        peak_p_score = max(peak_p_score, max_p)
        
        comp = get_complexity(roi_gray)
        # Track frame with best clarity (least likely to be motion blurred)
        if best_roi_gray is None or comp > get_complexity(best_roi_gray):
            best_roi_gray = roi_gray
            
        if max_p < 0.75 and comp > 200:
            frame_candidates.append({'gray': roi_gray, 'comp': comp})

    # 2. LAW: PLAYER-FIRST FORENSICS
    # If the player is detected, we check background visibility BEFORE guessing ores.
    if peak_p_score > PLAYER_PRESENCE_TRESHOLD and best_roi_gray is not None:
        val, is_empty = check_side_slice_empty(best_roi_gray)
        if is_empty: return "likely_empty", round(val, 4), 0, peak_p_score, "[L]"

    # 3. Trinity Identification Logic
    tier_z_momentum