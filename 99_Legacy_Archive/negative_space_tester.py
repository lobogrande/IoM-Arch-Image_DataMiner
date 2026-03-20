import cv2
import numpy as np
import os
import json

# --- TARGET CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 18 # Your provided problem floor
UNIFIED_ROOT = "Unified_Consensus_Inputs"
AI_DIM = 48
SLOT1_CENTER = (74, 261)
STEP_X, STEP_Y = 59.1, 59.1

def run_negative_space_lab():
    t_path = "templates"
    bg_templates = [cv2.imread(os.path.join(t_path, f), 0) for f in os.listdir(t_path) if f.startswith("background")]
    
    # Ensure all are 48x48
    bg_templates = [cv2.resize(t, (AI_DIM, AI_DIM)) for t in bg_templates if t is not None]
    
    print(f"--- Negative Space Lab: Testing {len(bg_templates)} BG Templates ---")

    run_path = os.path.join(UNIFIED_ROOT, f"Run_{TARGET_RUN}")
    with open(os.path.join(run_path, "final_sequence.json"), 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}

    img = cv2.imread(os.path.join(run_path, f"F{TARGET_FLOOR}_{sequence[TARGET_FLOOR]['frame']}"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(SLOT1_CENTER[0] + (col * STEP_X)), int(SLOT1_CENTER[1] + (row * STEP_Y))
        x1, y1, x2, y2 = cx-24, cy-24, cx+24, cy+24
        roi = gray[y1:y2, x1:x2]

        # Find the BEST background match
        scores = [cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max() for t in bg_templates]
        best_bg_score = max(scores) if scores else 0

        # Draw Logic
        # If score is HIGH, it's definitely empty.
        # If score is LOW, it might be an ore (or we need better BG templates)
        if best_bg_score > 0.70:
            color = (255, 0, 0) # Blue = Confirmed Empty
            label = f"BG:{best_bg_score:.2f}"
        else:
            color = (0, 0, 255) # Red = Potential Ore / Unknown
            label = f"UNK:{best_bg_score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, label, (x1+2, y2-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    out_name = f"Lab_F{TARGET_FLOOR}_Diagnostic.jpg"
    cv2.imwrite(out_name, img)
    print(f"Generated {out_name}. Check the scores on the empty slots.")

if __name__ == "__main__":
    run_negative_space_lab()