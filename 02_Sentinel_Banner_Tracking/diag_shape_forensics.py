# diag_shape_forensics.py
# Purpose: Analyze physical silhouettes of blocks to verify morphological consistency across all tiers.
# Version: 1.2 (Clipping Detection & Stability Check)

import sys, os, cv2, numpy as np, pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

# Expanded targets to map the "Geometric Soul" of every block family
TARGET_FRAMES = {
    # TIER: [Frame, Slot, State]
    'dirt1': [[121, 0, 'act'], [122, 0, 'sha']],
    'dirt2': [[264, 1, 'act'], [265, 1, 'sha']],
    'com1':  [[1, 3, 'act'],   [10, 3, 'sha']],
    'leg1':  [[847, 2, 'act'], [848, 2, 'sha']],
    'myth2': [[5073, 2, 'act'], [5074, 2, 'sha']],
    'epic1': [[264, 3, 'act'], [265, 3, 'sha']]
}

ORE0_X, ORE0_Y = 72, 255
STEP = 59.0
DIM = int(48 * 1.20)
ROW4_Y = int(ORE0_Y + (3 * STEP)) + 2

def get_silhouette(roi_gray):
    """Extracts a binary mask of the block using adaptive thresholding."""
    # Light blur to merge small noise into the background
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    
    # Otsu thresholding to find the strongest separation point
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check central region to determine if we need to invert (ensure block is white)
    h, w = mask.shape
    center_region = mask[h//3:2*h//3, w//3:2*w//3]
    if np.mean(center_region) < 127:
        mask = cv2.bitwise_not(mask)
        
    return mask

def analyze_shape(mask):
    """Calculates geometric properties of the binary silhouette and checks for clipping."""
    h, w = mask.shape
    max_area = h * w
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    # Pick the largest blob (the block)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    
    # Logic: If the area is nearly the whole box, the sensor 'clipped' to the boundary
    clipping = area > (max_area * 0.95)
    
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    
    # Log scale Hu moments for stability/readability
    hu_log = -np.sign(hu) * np.log10(np.abs(hu)) if np.any(hu) else [0.0]*7
    
    return {
        'area': area,
        'hu1': hu_log[0],
        'clipping': clipping
    }

def run_shape_audit():
    buffer_dir = cfg.get_buffer_path(0)
    if not os.path.exists(buffer_dir):
        print(f"Error: Buffer not found at {buffer_dir}")
        return

    all_files = sorted([f for f in os.listdir(buffer_dir) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- COMPREHENSIVE MORPHOLOGICAL AUDIT v1.2 ---")
    
    all_stats = []
    for tier, pairs in TARGET_FRAMES.items():
        results = []
        print(f"\nAnalyzing Tier: {tier.upper()}")
        
        for f_idx, col, state in pairs:
            if f_idx >= len(all_files): continue
            img = cv2.imread(os.path.join(buffer_dir, all_files[f_idx]), 0)
            if img is None: continue
            
            cx = int(ORE0_X + (col * STEP))
            x1, y1 = int(cx - DIM//2), int(ROW4_Y - DIM//2)
            roi = img[y1:y1+DIM, x1:x1+DIM]
            
            mask = get_silhouette(roi)
            stats = analyze_shape(mask)
            
            if stats:
                clip_tag = " [CLIPPED]" if stats['clipping'] else ""
                print(f"  {state.upper()}: Area={stats['area']:.1f}, Hu1={stats['hu1']:.4f}{clip_tag}")
                results.append(stats)

        if len(results) == 2:
            # We only track consistency if at least one side isn't clipped
            if not (results[0]['clipping'] and results[1]['clipping']):
                area_diff = abs(results[0]['area'] - results[1]['area']) / max(1, results[0]['area'])
                hu_diff = abs(results[0]['hu1'] - results[1]['hu1'])
                print(f"  >> Consistency: Area Var={area_diff*100:.1f}%, Hu1 Delta={hu_diff:.4f}")
                all_stats.append({
                    'tier': tier,
                    'area_var': round(area_diff, 3),
                    'hu_delta': round(hu_diff, 4)
                })
            else:
                print(f"  >> Consistency: [INCONCLUSIVE] Both states clipped to boundary.")

    if all_stats:
        df = pd.DataFrame(all_stats)
        print(f"\n--- FINAL FORENSIC SUMMARY ---")
        print(f"Mean Hu1 Delta (Valid Shapes): {df['hu_delta'].mean():.4f}")
        print(f"Most Geometric Consistent Tier: {df.loc[df['hu_delta'].idxmin(), 'tier']}")
    else:
        print("\n--- FINAL FORENSIC SUMMARY ---")
        print("No valid non-clipped shapes found for consistency tracking.")

if __name__ == "__main__":
    run_shape_audit()