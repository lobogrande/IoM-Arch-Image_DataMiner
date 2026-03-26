import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/VisualIndex"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_visual_indexer():
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    # We will create "Contact Sheets" of 100 frames each (10x10 grid)
    frames_per_sheet = 100
    grid_size = 10
    thumb_w, thumb_h = 120, 150 # Small thumbnails for speed

    print(f"--- Running Visual Indexer ---")
    print(f"Generating contact sheets for {len(files)} frames...")

    for sheet_idx in range(0, len(files), frames_per_sheet):
        # Create a blank canvas for the 10x10 grid
        canvas = np.zeros((thumb_h * grid_size, thumb_w * grid_size, 3), dtype=np.uint8)
        
        for i in range(frames_per_sheet):
            file_idx = sheet_idx + i
            if file_idx >= len(files): break
            
            # Load and resize
            img = cv2.imread(os.path.join(BUFFER_ROOT, files[file_idx]))
            if img is None: continue
            
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            
            # Label the thumbnail with its INDEX
            cv2.rectangle(thumb, (0, 0), (60, 25), (0, 0, 0), -1)
            cv2.putText(thumb, str(file_idx), (5, 18), 0, 0.5, (0, 255, 0), 1)
            
            # Place in grid
            row = i // grid_size
            col = i % grid_size
            canvas[row*thumb_h:(row+1)*thumb_h, col*thumb_w:(col+1)*thumb_w] = thumb

        # Save the sheet
        out_name = f"Index_Batch_{sheet_idx:05}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), canvas)
        print(f" [OK] Saved {out_name}", end='\r')

    print(f"\n[FINISH] Sheets saved to {OUTPUT_DIR}. Browse these to find your start_idx.")

if __name__ == "__main__":
    run_visual_indexer()