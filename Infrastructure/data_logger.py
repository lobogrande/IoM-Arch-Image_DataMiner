import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import mss
import pytesseract
import time
import csv
import os
from datetime import datetime

# --- GLOBAL SCALER (SET TO 2 FOR MAC RETINA, 1 FOR STANDARD) ---
SCALER = 2 
LOG_DIR = cfg.DATA_DIRS["LOGS"]

def scale_roi(roi):
    return {k: v * SCALER for k, v in roi.items()}

def scale_coords(coords):
    return [(x * SCALER, y * SCALER) for x, y in coords]

# --- VERIFIED COORDINATES (Original Point Values) ---
GAME_ROI_PT = {'top': 225, 'left': 10, 'width': 446, 'height': 677}
HEADER_ROI_PT = {'top': 282, 'left': 113, 'width': 26, 'height': 15}

SLOT_COORDS_PT = [
    (82, 492), (141, 492), (200, 492), (259, 492), (318, 492), (377, 492),
    (82, 551), (141, 551), (200, 551), (259, 551), (318, 551), (377, 551),
    (82, 610), (141, 610), (200, 610), (259, 610), (318, 610), (377, 610),
    (82, 669), (141, 669), (200, 669), (259, 669), (318, 669), (377, 669)
]

# Apply Global Scaling
GAME_ROI = scale_roi(GAME_ROI_PT)
HEADER_ROI = scale_roi(HEADER_ROI_PT)
SLOT_COORDS = scale_coords(SLOT_COORDS_PT)

# --- CONFIGURATION & THRESHOLDS ---
TESS_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789'
CSV_FILE = "high_speed_minigame_data.csv"
CHECK_FOLDER = "check_me"

ORE_THRESH = 0.82
BG_THRESH = 0.90  
FAIRY_THRESH = 0.85
PLAY_THRESH = 0.75 
SHA_THRESH = 0.75      # Lowered for better kill detection
VALIDATION_MIN = 0.70  # Threshold to save unknown images for review

if not os.path.exists(CHECK_FOLDER): os.makedirs(CHECK_FOLDER)

# --- TEMPLATE LOADING ---
def load_templates(folder=cfg.TEMPLATE_DIR):
    templates = {'act': {}, 'sha': {}, 'hbar': {}, 'bg': [], 'fairy': [], 'play': {}}
    if not os.path.exists(folder):
        print(f"CRITICAL ERROR: '{folder}' folder not found!")
        return templates
        
    for file in os.listdir(folder):
        if not file.endswith(".png"): continue
        img = cv2.imread(os.path.join(folder, file), 0)
        if img is None: continue
        
        if 'background_plain' in file:
            templates['bg'].append(img)
        elif 'xhair_fairy' in file:
            templates['fairy'].append(img)
        else:
            prefix = file.split('_')[0]
            if '_play' in file:
                if prefix not in templates['play']: templates['play'][prefix] = []
                templates['play'][prefix].append(img)
            elif 'act' in file:
                if 'hbar' in file: templates['hbar'][prefix] = img
                else:
                    if prefix not in templates['act']: templates['act'][prefix] = []
                    templates['act'][prefix].append(img)
            elif 'sha' in file:
                if prefix not in templates['sha']: templates['sha'][prefix] = []
                templates['sha'][prefix].append(img)
    return templates

all_templates = load_templates()

# --- STATE TRACKING ---
class SlotTracker:
    def __init__(self, index):
        self.index = index
        self.state = "EMPTY"
        self.tier = None
        self.hits = 0
        self.modifier = "None"
        self.debug_color = (0, 0, 0)

slots = [SlotTracker(i) for i in range(24)]
current_floor = "0"
board_is_active = False

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Floor", "Tier", "HitCount", "Modifier"])

print("--- PRO LOGGER v4.0 (RETINA OPTIMIZED) ---")
print(f"Monitoring 6x4 Grid | Scaler: {SCALER}x")

with mss.mss() as sct:
    while True:
        loop_start = time.time()
        screen = np.array(sct.grab(GAME_ROI))
        gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
        
        # 6x4 Grid visual (scaled for visibility)
        debug_canvas = np.zeros((200, 300, 3), dtype=np.uint8)

        # 1. GLOBAL FAIRY SCAN
        fairy_pos = None
        for f_temp in all_templates['fairy']:
            res = cv2.matchTemplate(gray, f_temp, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > FAIRY_THRESH:
                fairy_pos = (max_loc[0] + (f_temp.shape[1]//2), max_loc[1] + (f_temp.shape[0]//2))
                break

        any_blocks_on_board = False

        # 2. PER-SLOT PROCESSING
        for i, (cx, cy) in enumerate(SLOT_COORDS):
            lx, ly = cx - GAME_ROI['left'], cy - GAME_ROI['top']
            s = slots[i]
            grid_r, grid_c = i // 6, i % 6

            # Fairy Suppression (Radius scaled by 2)
            if fairy_pos:
                dist = np.sqrt((lx - fairy_pos[0])**2 + (ly - fairy_pos[1])**2)
                if dist < (45 * SCALER):
                    s.debug_color = (255, 0, 0) # Blue
                    continue 

            # Cell dimensions also scaled by 2
            cell_size = 25 * SCALER
            cell = gray[max(0, ly-cell_size):ly+cell_size, max(0, lx-cell_size):lx+cell_size]
            if cell.size == 0: continue

            # Baseline Check
            is_bg = False
            for bg_temp in all_templates['bg']:
                if np.max(cv2.matchTemplate(cell, bg_temp, cv2.TM_CCOEFF_NORMED)) > BG_THRESH:
                    if s.state == "ACTIVE": s.state = "EMPTY" # Transition back to empty
                    s.debug_color = (0, 0, 0)
                    is_bg = True
                    break
            if is_bg: continue

            # Player Overlap Check
            for tier, t_imgs in all_templates['play'].items():
                for t_img in t_imgs:
                    if np.max(cv2.matchTemplate(cell, t_img, cv2.TM_CCOEFF_NORMED)) > PLAY_THRESH:
                        s.modifier = "PLAYER_OVERLAP"
                        s.debug_color = (255, 128, 0) # Orange
                        break

            # Block Detection
            found_active = False
            max_conf = 0
            for tier, t_imgs in all_templates['act'].items():
                for t_img in t_imgs:
                    res = cv2.matchTemplate(cell, t_img, cv2.TM_CCOEFF_NORMED)
                    conf = np.max(res)
                    if conf > ORE_THRESH:
                        if s.state == "EMPTY":
                            s.state = "ACTIVE"
                            s.tier = tier
                            s.hits = 0
                        s.debug_color = (0, 255, 0) # Green
                        found_active = True
                        any_blocks_on_board = True
                        
                        # Hit Detection
                        if tier in all_templates['hbar']:
                            if np.max(cv2.matchTemplate(cell, all_templates['hbar'][tier], cv2.TM_CCOEFF_NORMED)) > ORE_THRESH:
                                s.hits += 1
                        break
                    elif conf > max_conf: max_conf = conf

            # Low Confidence Capture
            if not found_active and VALIDATION_MIN < max_conf < ORE_THRESH:
                cv2.imwrite(f"{CHECK_FOLDER}/low_conf_{int(time.time()*1000)}.png", cell)

            # Kill Trigger (Sha Check + BG Fallback)
            if s.state == "ACTIVE" and not found_active:
                killed = False
                # Try Shadow Match first
                for tier, t_imgs in all_templates['sha'].items():
                    for t_img in t_imgs:
                        if np.max(cv2.matchTemplate(cell, t_img, cv2.TM_CCOEFF_NORMED)) > SHA_THRESH:
                            killed = True
                            break
                
                # Fallback: If it's just Background now, it counts as a kill
                if not killed:
                    for bg_temp in all_templates['bg']:
                        if np.max(cv2.matchTemplate(cell, bg_temp, cv2.TM_CCOEFF_NORMED)) > BG_THRESH:
                            killed = True
                            s.modifier += "_FALLBACK"
                            break

                if killed:
                    with open(CSV_FILE, 'a', newline='') as f:
                        csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S.%f"), current_floor, s.tier, s.hits, s.modifier])
                    print(f"\n[KILL] Floor {current_floor} | Tier: {s.tier} | Hits: {s.hits} | Mod: {s.modifier}")
                    s.state = "EMPTY"
                    s.modifier = "None"

            # Draw to Debug Canvas
            cv2.rectangle(debug_canvas, (grid_c*50, grid_r*50), (grid_c*50+48, grid_r*50+48), s.debug_color, -1)

        # 3. HIGH-SPEED OCR
        if any_blocks_on_board and not board_is_active:
            h_cap = np.array(sct.grab(HEADER_ROI))
            h_inv = cv2.bitwise_not(cv2.cvtColor(h_cap, cv2.COLOR_BGRA2GRAY))
            current_floor = pytesseract.image_to_string(h_inv, config=TESS_CONFIG).strip()
            board_is_active = True
        elif not any_blocks_on_board:
            board_is_active = False

        cv2.imshow("Board Status (6x4)", debug_canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        print(f"\rFloor: {current_floor} | Loop: {(time.time()-loop_start)*1000:.1f}ms", end="")