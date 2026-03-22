# diag_shadow_signal_forensics.py
# Purpose: Isolate the "Latent Signal" in shadow ores to stop identity-flipping.
# Version: 1.0 (Differential Analysis Pass)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# TARGET: Frame 73 is our "Ground Zero" for shadow failure
TARGET_FRAME = 73
# We know Slot 5 is a clean Active Rare1. Use it as the anchor.
ANCHOR_IDENTITY = "rare1" 
SHADOW_SLOTS = [0, 1, 3]

ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
SIDE_PX = int(48 * 1.20)
ROW4_Y = int(ORE0_Y + (3 * STEP)) + 2

def get_roi(img, col):
    cx = int(ORE0_X + (col * STEP))
    x1, y1 = int(cx - SIDE_PX//2), int(ROW4_Y - SIDE_PX//2)
    return img[y1:y1+SIDE_PX, x1:x1+SIDE_PX]

def apply_pipelines(roi):
    """Generates 4 distinct visual interpretations of the ore."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape)==3 else roi
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # 1. CLAHE (Current)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(norm)
    # 2. Sobel (Gradient Direction)
    grad_x = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.sqrt(cv2.add(cv2.pow(grad_x, 2), cv2.pow(grad_y, 2))))
    # 3. Laplacian (Grain Density)
    lap = cv2.convertScaleAbs(cv2.Laplacian(norm, cv2.CV_64F))
    # 4. Binary Silhouette (Shape)
    _, sil = cv2.threshold(cv2.GaussianBlur(norm, (5,5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return {'clahe': clahe, 'grad': grad, 'lap': lap, 'sil': sil}

def run_forensic_audit():
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    img = cv2.imread(os.path.join(buffer_dir, all_files[TARGET_FRAME]))
    
    # Load Templates for Comparison
    t_path = cfg.TEMPLATE_DIR
    templates = {
        'rare1_sha': cv2.imread(os.path.join(t_path, "rare1_sha_plain_0.png"), 0),
        'dirt1_sha': cv2.imread(os.path.join(t_path, "dirt1_sha_plain_0.png"), 0),
        'com1_act':  cv2.imread(os.path.join(t_path, "com1_act_plain_0.png"), 0)
    }
    # Scale templates
    for k in templates: 
        templates[k] = cv2.resize(templates[k], (SIDE_PX, SIDE_PX))
        templates[k] = apply_pipelines(templates[k]) # Process templates same as ROI

    print(f"--- SHADOW SIGNAL FORENSICS: FRAME {TARGET_FRAME} ---")
    print(f"ANCHOR: {ANCHOR_IDENTITY} (Slot 5)\n")

    results = []
    for col in SHADOW_SLOTS:
        roi = get_roi(img, col)
        roi_pipes = apply_pipelines(roi)
        
        print(f"Slot {col} Shadow Analysis:")
        for pipe_name, roi_proc in roi_pipes.items():
            # Compare processed ROI against Rare1 (Truth) and Dirt1 (Current Error)
            res_rare = cv2.matchTemplate(roi_proc, templates['rare1_sha'][pipe_name], cv2.TM_CCOEFF_NORMED)
            res_dirt = cv2.matchTemplate(roi_proc, templates['dirt1_sha'][pipe_name], cv2.TM_CCOEFF_NORMED)
            
            score_rare = cv2.minMaxLoc(res_rare)[1]
            score_dirt = cv2.minMaxLoc(res_dirt)[1]
            
            winner = "RARE1" if score_rare > score_dirt else "DIRT1 (Error)"
            gap = abs(score_rare - score_dirt)
            print(f"  [{pipe_name.upper()}] Rare1:{score_rare:.3f} vs Dirt1:{score_dirt:.3f} | Winner: {winner} (Gap:{gap:.3f})")
            
            results.append({'slot': col, 'pipe': pipe_name, 'rare': score_rare, 'dirt': score_dirt, 'gap': gap})

    print(f"\n--- CONCLUSION ---")
    df = pd.DataFrame(results)
    best_pipe = df.groupby('pipe')['gap'].mean().idxmax()
    print(f"The most discriminatory pipeline for this row is: {best_pipe.upper()}")
    print("Next Step: Integrate this pipeline into the shadow-state detection loop.")

if __name__ == "__main__":
    run_forensic_audit()