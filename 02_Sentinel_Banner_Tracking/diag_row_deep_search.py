# diag_row_deep_search.py
# Purpose: Deep dive into a specific row to see candidate rankings and scores.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# Problematic frames from your feedback
TARGET_FRAMES = [63, 64, 519]

ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
DIM_ID = 48
TARGET_SCALE = 1.20
SIDE_PX = int(DIM_ID * TARGET_SCALE)

def apply_gamma_lift(img, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def run_deep_search():
    buffer_dir = cfg.get_buffer_path(0)
    t_path = cfg.TEMPLATE_DIR
    
    # Load all plain templates
    templates = []
    for f in os.listdir(t_path):
        if "_plain_" in f and not any(x in f for x in ["background", "player", "negative"]):
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None:
                img_scaled = cv2.resize(img, (SIDE_PX, SIDE_PX))
                templates.append({'name': f.split("_")[0], 'img': img_scaled, 'filename': f})

    print(f"--- DEEP SEARCH FORENSICS ---")
    
    for f_idx in TARGET_FRAMES:
        all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        row4_y = int(ORE0_Y + (3 * STEP)) + 2
        
        print(f"\nFRAME {f_idx}:")
        for col in range(6):
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - SIDE_PX//2), int(row4_y - SIDE_PX//2)
            roi = img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]
            roi_lifted = apply_gamma_lift(roi, 0.5)
            
            results = []
            for tpl in templates:
                res = cv2.matchTemplate(roi_lifted, tpl['img'], cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                results.append({'tier': tpl['name'], 'score': score, 'file': tpl['filename']})
            
            df = pd.DataFrame(results).sort_values('score', ascending=False).head(5)
            print(f"  Slot {col} Top 5 Candidates:")
            for _, r in df.iterrows():
                print(f"    {r['tier']}: {r['score']:.4f} ({r['file']})")

if __name__ == "__main__":
    run_deep_search()