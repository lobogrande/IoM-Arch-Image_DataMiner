# grid_ground_truth_audit.py
# Purpose: Derive the true grid constants (STEP_X, S0_X, S0_Y) by mapping block centers on Frame 0.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

DIAG_OUT = "grid_ground_truth_verify.jpg"

def run_block_audit():
    print("--- STARTING ORE GRID GROUND-TRUTH AUDIT ---")
    
    # 1. Load active "Dirt1" template (we assume this is standard Tier 1)
    # The template name 'act_dirt1.png' is extrapolated fromreference materials
    block_t = cv2.imread(os.path.join(cfg.TEMPLATE_DIR, "act_dirt1.png"), 0)
    if block_t is None: 
        print("Error: act_dirt1.png not found. Cannot perform block grid audit.")
        return
    oh, ow = block_t.shape

    files = sorted([f for f in os.listdir(cfg.get_buffer_path(0)) if f.endswith('.png')])
    frame_0 = files[0] # Use Frame 0 where all blocks are active

    print(f"Analyzing Frame 0 ({frame_0}) for Dirt1 Blocks...")
    img = cv2.imread(os.path.join(cfg.get_buffer_path(0), frame_0), 0)
    if img is None: return
    ih, iw = img.shape
    
    # Define a generous scan strip for Row 1
    y1, y2 = 200, 350
    strip = img[y1:y2, :]

    # Global matching for all instances of Tier 1 Block
    res = cv2.matchTemplate(strip, block_t, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    locs = np.where(res >= threshold)
    
    # Convert matches to centers, adjusting for strip offset
    matches = []
    for pt in zip(*locs[::-1]):
        matches.append((pt[0] + (ow // 2), pt[1] + y1 + (oh // 2)))

    # Deduplicate matches found too close to each other (use 30px proximity)
    deduped = []
    if matches:
        matches.sort() # Sort by X
        current_x, current_y = matches[0]
        deduped.append((current_x, current_y))
        for i in range(1, len(matches)):
            next_x, next_y = matches[i]
            if next_x > current_x + 30: # If more than 30px away, it's a new block
                deduped.append((next_x, next_y))
                current_x, current_y = next_x, next_y

    print(f"Found {len(deduped)} potential blocks in Row 1.")
    
    # Create Verification Plot
    proof = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(proof, (0, y1), (iw, y2), (255, 255, 0), 2) # Draw the scan region
    
    final_centers = []
    # If we found at least two blocks, calculate the steps
    if len(deduped) >= 3:
        # Sort and label the first three (0, 1, 2)
        deduped.sort()
        
        # ORE 0: PIXEL GROUND TRUTH
        block0_x, block0_y = deduped[0]
        block1_x, block1_y = deduped[1]
        block2_x, block2_y = deduped[2]
        
        step_x_01 = block1_x - block0_x
        step_x_12 = block2_x - block1_x
        
        print("\n--- GRIND TRUTH RESULTS ---")
        print(f"Block 0 Center (AI Pixel): ({ore0_x}, {ore0_y})")
        print(f"Block 1 Center (AI Pixel): ({ore1_x}, {ore1_y})")
        print(f"Block 2 Center (AI Pixel): ({ore2_x}, {ore2_y})")
        print(f"Measured STEP_X (0->1): {step_x_01}")
        print(f"Measured STEP_X (1->2): {step_x_12}")
        
        # Apply visual labels
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # R, G, B
        for i in range(min(3, len(deduped))):
            x, y = deduped[i]
            cv2.circle(proof, (x, y), 5, colors[i], -1)
            cv2.putText(proof, f"Block {i} Center: ({x}, {y})", (x, y - 20), 0, 0.5, colors[i], 1)

        cv2.imwrite(DIAG_OUT, proof)
        print(f"Verification image saved: {DIAG_OUT}")
        
    else:
        print("\n[ERROR] Failed to find ground-truth blocks. Cannot finalize grid.")

if __name__ == "__main__":
    run_block_audit()