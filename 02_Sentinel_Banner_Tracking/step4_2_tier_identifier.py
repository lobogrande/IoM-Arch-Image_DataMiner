# step4_2_tier_identifier.py
# Purpose: Master Plan Step 4.2 - Identify ore tiers using Temporal Consensus 
#          with CLAHE contrast enhancement and Side-Slice Forensics.
# Version: 1.7 (The High-Contrast Forensic Scalpel)

import sys, os, cv2, numpy as np, pandas as pd
import concurrent.futures
from functools import partial
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# INPUT/OUTPUT
BOUNDARIES_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "final_floor_boundaries.csv")
DNA_INVENTORY_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_dna_inventory.csv")
OUT_CSV = os.path.join(cfg.DATA_DIRS["TRACKING"], "floor_ore_inventory.csv")
VERIFY_DIR = os.path.join(cfg.DATA_DIRS["TRACKING"], "ore_identification_proofs")

# DIAGNOSTIC CONTROL
LIMIT_FLOORS = 20  

# GRID CONSTANTS
ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SCALE = 1.20
SIDE_PX = int(48 * SCALE) # 57px
HUD_DX, HUD_DY = 20, 30

# THRESHOLDS
MIN_MATCH_CONFIDENCE = 0.45  
PLAYER_REJECTION_GATE = 0.75 
SIDE_SLICE_GATE = 0.70       # Lowered for thin vertical strip reliability
HARVEST_COUNT = 15          
RARITY_BIAS = 0.05           # Extra margin given to high-tiers to prevent Dirt-Collisions

def apply_clahe(img):
    """Enhances local contrast to reveal subtle ore facets."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def load_resources():
    """Pre-loads ores, players, and background templates."""
    res = {'ores': {}, 'player': [], 'bg': []}
    t_path = cfg.TEMPLATE_DIR
    for f in os.listdir(t_path):
        img = cv2.imread(os.path.join(t_path, f), 0)
        if img is None: continue
        img_48 = cv2.resize(img, (48, 48))
        if "_act_plain_" in f and not any(x in f for x in ["player", "background"]):
            tier = f.split("_")[0]
            if tier not in res['ores']: res['ores'][tier] = []
            # Templates also get CLAHE to match the processed ROI
            res['ores'][tier].append(apply_clahe(img_48))
        if "negative_player" in f: res['player'].append(img_48)
        if "background_plain" in f: res['bg'].append(img_48)
    return res

def check_side_slice_empty(roi_gray, bg_tpls, is_banner):
    """Forensic check: Does the left side of the slot match the background?"""
    # 30x30 central ore ROI
    roi_30 = roi_gray[13:43, 13:43]
    if is_banner: roi_30 = roi_30[12:, :]
    
    # Peek at the left 6 pixels (tighter to avoid player sprite arms/pickaxe)
    slice_roi = apply_clahe(roi_30[:, 0:6])
    best_s = 0
    for tpl in bg_tpls:
        tpl_30 = tpl[9:39, 9:39]
        if is_banner: tpl_30 = tpl_30[12:, :]
        slice_tpl = apply_clahe(tpl_30[:, 0:6])
        res = cv2.matchTemplate(slice_tpl, slice_roi, cv2.TM_CCOEFF_NORMED)
        best_s = max(best_s, cv2.minMaxLoc(res)[1])
    return best_s > SIDE_SLICE_GATE

def identify_consensus(f_range, r_idx, col_