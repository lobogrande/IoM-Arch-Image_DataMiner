# diag_step2_dna.py
# Purpose: Deep dive into false DNA shifts caused by transparent particles/smoke.
# Target: Run 1, Frame 24300 - 24370, Row 3 Slot 0.

import sys, os, cv2, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# --- TARGET CONFIG ---
TARGET_RUN = "4"
START_FRAME_IDX = 24300
END_FRAME_IDX = 24370
# Row 3, Slot 0
TARGET_ROW = 2 
TARGET_COL = 0 

ORE0_X, ORE0_Y = 74, 261
STEP = 59.0

def run_dna_diagnostic():
    buffer_dir = cfg.get_buffer_path(TARGET_RUN)
    out_dir = f"diagnostic_dna_r{TARGET_ROW+1}_s{TARGET_COL}"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])

    # Load Background Templates
    bg_tpls =[]
    for i in range(10):
        p = os.path.join(cfg.TEMPLATE_DIR, f"background_plain_{i}.png")
        if os.path.exists(p): bg_tpls.append(cv2.imread(p, 0))

    print(f"--- DNA SENSOR FORENSICS (Run {TARGET_RUN}) ---")
    print(f"Target: Row {TARGET_ROW+1} Slot {TARGET_COL} | Frames {START_FRAME_IDX} to {END_FRAME_IDX}")
    print(f"{'Frame':<8} | {'Best BG Score':<15} | {'Status'}")
    print("-" * 40)

    y_center = int(ORE0_Y + (TARGET_ROW * STEP))
    x_center = int(ORE0_X + (TARGET_COL * STEP))
    tx, ty = x_center - 15, y_center - 15

    for i in range(START_FRAME_IDX, END_FRAME_IDX + 1):
        if i >= len(all_files): break
        img_bgr = cv2.imread(os.path.join(buffer_dir, all_files[i]))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        roi = img_gray[ty:ty+30, tx:tx+30]
        
        # Calculate best BG match
        best_score = 0.0
        for tpl in bg_tpls:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(res)[1]
            if score > best_score: best_score = score

        # Formatting Output
        status = "EMPTY (0)" if best_score >= 0.75 else "OCCUPIED (1) <-- FALSE FLIP"
        
        # If it drops below 0.85, save a zoomed-in image and apply histogram equalization to reveal the smoke
        if best_score < 0.85:
            roi_zoom = cv2.resize(roi, (120, 120), interpolation=cv2.INTER_NEAREST)
            # Enhance contrast to reveal invisible particles
            roi_enhanced = cv2.equalizeHist(roi_zoom)
            
            # Combine raw and enhanced side-by-side
            combined = np.hstack((roi_zoom, roi_enhanced))
            
            # Add text
            cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR, combined)
            cv2.putText(combined, f"Score: {best_score:.3f}", (5, 15), 0, 0.4, (255,255,255), 1)
            
            cv2.imwrite(os.path.join(out_dir, f"frame_{i}_score_{best_score:.3f}.png"), combined)

        print(f"{i:<8} | {best_score:<15.4f} | {status}")

if __name__ == "__main__":
    run_dna_diagnostic()