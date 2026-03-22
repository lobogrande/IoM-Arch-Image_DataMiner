# diag_shape_forensics.py
# Purpose: Analyze physical silhouettes of ores to verify morphological consistency.

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

TARGET_FRAMES = {
    5073: [2], # Myth2 Active
    5074: [2]  # Myth2 Shadow/Hit
}

ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
DIM = int(48 * 1.20)
ROW4_Y = int(ORE0_Y + (3 * STEP)) + 2

def get_silhouette(roi_gray):
    """Extracts a binary mask of the ore using adaptive thresholding."""
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    # Threshold to find the central 'blob'
    # For active: it's the bright part. For shadow: it's the dark part.
    # We use Otsu's to find the best separation.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If the center pixel is black, we might need to invert (shadow state)
    if mask[mask.shape[0]//2, mask.shape[1]//2] == 0:
        mask = cv2.bitwise_not(mask)
        
    return mask

def analyze_shape(mask):
    """Calculates geometric properties of the binary silhouette."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    # Log scale Hu moments for readability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu))
    
    return {
        'area': cv2.contourArea(cnt),
        'perimeter': cv2.arcLength(cnt, True),
        'hu': hu_log
    }

def run_shape_audit():
    buffer_dir = cfg.get_buffer_path(0)
    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    results = []
    for f_idx, slots in TARGET_FRAMES.items():
        if f_idx >= len(all_files): continue
        img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
        
        for col in slots:
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - DIM//2), int(ROW4_Y - DIM//2)
            roi = img[y1:y1+DIM, x1:x1+DIM]
            
            mask = get_silhouette(roi)
            stats = analyze_shape(mask)
            
            if stats:
                print(f"F{f_idx} S{col}: Area={stats['area']:.1f}, Hu1={stats['hu'][0]:.4f}")
                results.append({'frame': f_idx, 'area': stats['area'], 'hu1': stats['hu'][0]})

    if len(results) >= 2:
        area_diff = abs(results[0]['area'] - results[1]['area']) / results[0]['area']
        hu_diff = abs(results[0]['hu1'] - results[1]['hu1'])
        print(f"\n--- MORPHOLOGICAL CONSISTENCY CHECK ---")
        print(f"Area Variance: {area_diff*100:.1f}%")
        print(f"Hu1 Similarity: {hu_diff:.4f} (Lower is more consistent)")

if __name__ == "__main__":
    run_shape_audit()