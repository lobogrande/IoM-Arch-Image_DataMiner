import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 7
UNIFIED_ROOT = "Unified_Consensus_Inputs"
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def run_heatmap_generator():
    # 1. Load Background Templates
    bg_templates = []
    t_path = "templates"
    for f in os.listdir(t_path):
        if f.startswith("background"):
            img = cv2.imread(os.path.join(t_path, f), 0)
            if img is not None: bg_templates.append(cv2.resize(img, (48, 48)))

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    files = [f for f in os.listdir(run_path) if f.startswith(f"F{TARGET_FLOOR}_")]
    raw_img = cv2.imread(os.path.join(run_path, files[0]))
    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    
    # This will store the 'heat' of the differences
    heatmap_canvas = np.zeros((raw_img.shape[0], raw_img.shape[1]), dtype=np.uint8)

    print(f"--- Visual Forensic Heatmap: Floor {TARGET_FLOOR} ---")

    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0]+(col*STEP_X)), int(SLOT1_CENTER[1]+(row*STEP_Y))
        x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
        roi = gray[y1:y2, x1:x2]
        
        # Find best background match and get the actual 'Difference Image'
        min_diff_img = None
        min_score = 999999
        
        for bg in bg_templates:
            diff = cv2.absdiff(roi, bg)
            score = np.sum(diff)
            if score < min_score:
                min_score = score
                min_diff_img = diff

        # Place the difference pixels onto the heatmap
        heatmap_canvas[y1:y2, x1:x2] = min_diff_img
        
        # Calculate readable score
        norm_score = min_score / (48*48)
        
        # DRAW HUD (White text with black outline)
        color = (255, 255, 255)
        text = f"D:{int(norm_score)}"
        # Outline for visibility
        cv2.putText(raw_img, text, (x1+2, y2-4), 0, 0.4, (0,0,0), 2)
        cv2.putText(raw_img, text, (x1+2, y2-4), 0, 0.4, color, 1)
        cv2.rectangle(raw_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Apply a color map to the heatmap to make it pop
    color_heatmap = cv2.applyColorMap(heatmap_canvas, cv2.COLORMAP_JET)
    
    cv2.imwrite(f"Heatmap_F{TARGET_FLOOR}_Visual.jpg", raw_img)
    cv2.imwrite(f"Heatmap_F{TARGET_FLOOR}_DifferenceMap.jpg", color_heatmap)
    print(f"Generated Heatmap_F{TARGET_FLOOR}_Visual.jpg and DifferenceMap.jpg")

if __name__ == "__main__":
    run_heatmap_generator()