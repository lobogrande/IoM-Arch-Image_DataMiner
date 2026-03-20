import cv2
import numpy as np
import os
import csv
import pytesseract
import sys

# --- 1. CONFIGURATION ---
BUFFER_DIR = "capture_buffer"
TEMPLATE_DIR = "templates"
RECOVERY_DIR = "forensic_verification_locked"
CSV_FILE = "FINAL_TOTAL_AUDIT_v31_44.csv"

START_FLOOR = 1
START_IMAGE_NAME = "frame_20260306_231742_176023.jpg"

# --- 2. BOSS DATA (Restored & Explicit) ---
BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'},
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'},
    34: {'tier': 'mixed', 'special': {8: 'myth1', 9: 'myth1', 14: 'myth1', 15: 'myth1',
                   **{i: 'com2' for i in range(24) if i not in [8, 9, 14, 15]}}},
    49: {'tier': 'mixed', 'special': {**{i: 'dirt3' for i in range(0, 6)}, **{i: 'com3' for i in range(6, 12)},
                   **{i: 'rare3' for i in range(12, 18)}, **{i: 'myth2' for i in range(18, 24)}}},
    74: {'tier': 'mixed', 'special': {20: 'div1', 21: 'div1', **{i: 'dirt3' for i in range(24) if i not in [20, 21]}}},
    98: {'tier': 'myth3'},
    99: {'tier': 'mixed', 'special': {**{i: 'com3' for i in [0, 6, 12, 18]}, **{i: 'rare3' for i in [1, 7, 13, 19]},
                   **{i: 'epic3' for i in [2, 8, 14, 20]}, **{i: 'leg3' for i in [3, 9, 15, 21]},
                   **{i: 'myth3' for i in [4, 10, 16, 22]}, **{i: 'div2' for i in [5, 11, 17, 23]}}}
}

# --- 3. DUAL-GRID MAPPING ---
# A. THE AI GRID (The "Golden" Search Pixels)
AI_SLOT_COORDS = {i: (100 + (i % 6) * 60, 500 + (i // 6) * 65) for i in range(24)}

# B. THE HUD GRID (Your calibrated clicker coordinates)
HUD_SLOT_COORDS = {i: (73 + (i % 6) * 59, 264 + (i // 6) * 61) for i in range(24)}

# C. The "Invisible" AI Search Area (Buffered for drift)
# We add a 20-pixel buffer to the Y-axis to catch Floor 14-style drift
AI_DIG_SEARCH_ROI = (211, 150, 50, 140) # (Y, X, H, W)

# D. The "Visible" HUD Box (Your exact clicks)
HUD_DIG_BOX_COORDS = (162, 231, 278, 244) # (X1, Y1, X2, Y2)

def detect_dig_text_refined(img):
    """AI-only sensor using the larger search buffer"""
    y, x, h, w = AI_DIG_SEARCH_ROI
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_pixels = np.sum(binary == 255)
    
    # We look for a 'pulse' of white text in this buffered zone
    return 20 < white_pixels < 400

# E. Header (Y, X, H, W)
HEADER_ROI = (56, 100, 16, 35)

# --- 4. ENGINE ---
def run_census_and_draw_hud(img_bgr, floor, boss_dict, templates):
    results = {}
    is_boss = floor in boss_dict
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hud_canvas = img_bgr.copy()

    for slot_id in range(24):
        # AI SEARCH (Uses Golden Coordinates)
        ax, ay = AI_SLOT_COORDS[slot_id]
        
        if is_boss:
            tier = boss_dict[floor]['special'].get(slot_id, boss_dict[floor]['tier']) if 'special' in boss_dict[floor] else boss_dict[floor]['tier']
            score, color = 1.0, (0, 255, 0) # Green for Boss
        else:
            slot_roi = gray_img[ay-24:ay+24, ax-24:ax+24]
            best_tier, best_score = "unknown", 0.0
            for t_name, t_img in templates.items():
                res = cv2.matchTemplate(slot_roi, t_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_score: best_score, best_tier = max_val, t_name
            
            # Increased threshold to 0.75 to stop "Divine" hallucinations on empty dirt
            tier, score = (best_tier, round(best_score, 2)) if best_score >= 0.75 else ("obscured", round(best_score, 2))
            color = (0, 255, 255) # Yellow for AI Match

        results[slot_id] = {'tier': tier, 'score': score}
        
        # HUD DRAW (Uses Calibrated Clicker Coordinates)
        hx, hy = HUD_SLOT_COORDS[slot_id]
        cv2.rectangle(hud_canvas, (hx-24, hy-24), (hx+24, hy+24), color, 1)
        cv2.putText(hud_canvas, f"{tier}:{score}", (hx-24, hy-28), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return results, hud_canvas

# --- 5. MASTER LOOP ---
def run_v31_44_audit():
    if not os.path.exists(RECOVERY_DIR): os.makedirs(RECOVERY_DIR)
    templates = {f.split('.')[0]: cv2.imread(os.path.join(TEMPLATE_DIR, f), 0) for f in os.listdir(TEMPLATE_DIR) if f.endswith('.png')}
    frames = sorted([f for f in os.listdir(BUFFER_DIR) if f.endswith(('.png', '.jpg'))])
    
    try: curr_ptr = frames.index(START_IMAGE_NAME)
    except: sys.exit("!!! Start Image Not Found")

    target_f, idx_end = START_FLOOR, len(frames) - 1
    
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Frame", "Floor", "Slot", "Tier", "Score", "Method"])

        while curr_ptr <= idx_end:
            print(f" Scanning {curr_ptr}/{idx_end} (Seeking F{target_f})...", end='\r')
            img = cv2.imread(os.path.join(BUFFER_DIR, frames[curr_ptr]))
            if img is None: (curr_ptr := curr_ptr + 1); continue
            
            # OCR Header Check
            roi_h = img[HEADER_ROI[0]:HEADER_ROI[0]+HEADER_ROI[2], HEADER_ROI[1]:HEADER_ROI[1]+HEADER_ROI[3]]
            text_h = pytesseract.image_to_string(roi_h, config='--psm 7 digits').strip()
            h_val = int(text_h) if text_h.isdigit() else -1

            # Dig Text Trigger
            roi_d = img[DIG_SITE_ROI[0]:DIG_SITE_ROI[0]+DIG_SITE_ROI[2], DIG_SITE_ROI[1]:DIG_SITE_ROI[1]+DIG_SITE_ROI[3]]
            gray_d = cv2.cvtColor(roi_d, cv2.COLOR_BGR2GRAY)
            _, bin_d = cv2.threshold(gray_d, 200, 255, cv2.THRESH_BINARY)
            is_dig = 20 < np.sum(bin_d == 255) < 300 

            # FIX: Only increment target_f if OCR actually confirms we moved forward
            if h_val > target_f and (h_val - target_f) <= 10: 
                target_f = h_val

            if h_val == target_f or is_dig:
                # Use h_val for the filename if available, else use internal target_f
                actual_f = h_val if h_val > 0 else target_f
                print(f"\n [FOUND] Floor {actual_f} at {frames[curr_ptr]}")

                census_ptr = min(curr_ptr + 6, idx_end)
                census_img = cv2.imread(os.path.join(BUFFER_DIR, frames[census_ptr]))
                
                ores, hud_img = run_census_and_draw_hud(census_img, actual_f, BOSS_DATA, templates)
                
                for slot, data in ores.items():
                    writer.writerow([frames[census_ptr], actual_f, slot, data['tier'], data['score'], "v31.44_Surgical"])
                
                cv2.rectangle(hud_img, (DIG_SITE_ROI[1], DIG_SITE_ROI[0]), (DIG_SITE_ROI[1]+DIG_SITE_ROI[3], DIG_SITE_ROI[0]+DIG_SITE_ROI[2]), (255, 0, 255), 1)
                cv2.imwrite(f"{RECOVERY_DIR}/Floor_{actual_f}_Verified.png", hud_img)
                
                target_f = actual_f + 1
                curr_ptr = census_ptr + 1
            else:
                curr_ptr += 1

if __name__ == "__main__":
    run_v31_44_audit()