import cv2
import numpy as np
import os
import pandas as pd

# --- CONFIGURATION ---
BUFFER_ROOT = "capture_buffer_0"
OUT_DIR = "sentinel_epsilon_debug"
TARGET_FILENAME = "frame_20260306_231817_939420.png"

# --- GEOMETRIC CONSTRAINTS (USER CALIBRATED) ---
SCAN_Y_START = 40   
SCAN_Y_END = 450     

# --- THRESHOLDS ---
VALLEY_THRESHOLD = 22.0  # Slightly relaxed to capture edges better
MIN_BANNER_H = 25        # Lowered to catch early nucleation
MAX_BANNER_H = 85
GAP_FILL = 40            # Bridge text noise

def detect_banner_epsilon(img_gray):
    h, w = img_gray.shape
    c1, c2 = int(w * 0.35), int(w * 0.65)
    center_strip = img_gray[:, c1:c2]
    intensities = np.mean(center_strip, axis=1)
    
    # Create mask within user-defined bounds
    mask = np.zeros(h, dtype=np.uint8)
    for y in range(SCAN_Y_START, SCAN_Y_END):
        if intensities[y] < VALLEY_THRESHOLD:
            mask[y] = 1
            
    # Bridge text gaps
    kernel = np.ones((GAP_FILL, 1), np.uint8)
    closed_mask = cv2.morphologyEx(mask.reshape(-1, 1), cv2.MORPH_CLOSE, kernel).flatten()
    
    zones = []
    start_y = None
    for y, is_active in enumerate(closed_mask):
        if is_active == 1 and start_y is None:
            start_y = y
        elif is_active == 0 and start_y is not None:
            height = y - start_y
            if MIN_BANNER_H <= height <= MAX_BANNER_H:
                # Check if it's NUC (Nucleation) or FULL (Full Width)
                full_row_avg = np.mean(img_gray[start_y:y, :])
                state = "NUC" if full_row_avg > 15.0 else "FULL"
                zones.append({'top': start_y, 'bot': y, 'state': state})
            start_y = None
    return zones, intensities

def draw_telemetry(img, intensities):
    h, w, _ = img.shape
    # Draw a small graph on the right side (width 100px)
    graph_x_start = w - 120
    cv2.rectangle(img, (graph_x_start, 0), (w, h), (30, 30, 30), -1)
    
    # Draw threshold line (Blue)
    thresh_x = int(graph_x_start + (VALLEY_THRESHOLD / 100.0) * 100)
    cv2.line(img, (thresh_x, SCAN_Y_START), (thresh_x, SCAN_Y_END), (255, 0, 0), 1)

    # Draw the intensity line
    for y in range(1, h):
        x1 = int(graph_x_start + (intensities[y-1] / 100.0) * 100)
        x2 = int(graph_x_start + (intensities[y] / 100.0) * 100)
        # Use Green if below threshold, Red if above
        color = (0, 255, 0) if intensities[y] < VALLEY_THRESHOLD else (0, 0, 255)
        cv2.line(img, (x1, y-1), (x2, y), color, 1)
        
    return img

def draw_pixel_ruler(img):
    for y in range(0, img.shape[0], 50):
        cv2.line(img, (0, y), (15, y), (200, 200, 200), 1)
        cv2.putText(img, str(y), (20, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    return img

def run_sentinel_epsilon():
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    all_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.lower().endswith(('.png', '.jpg'))])
    
    try: target_idx = all_files.index(TARGET_FILENAME)
    except: target_idx = 1170

    manifest = []
    # Analyzing the range from early nucleation to full exit
    for i in range(target_idx - 15, target_idx + 60):
        fname = all_files[i]
        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, fname))
        if img_bgr is None: continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        zones, intensities = detect_banner_epsilon(img_gray)
        
        # Overlay Visuals
        overlay = img_bgr.copy()
        for z in zones:
            color = (0, 255, 255) if z['state'] == "NUC" else (0, 0, 255)
            cv2.rectangle(overlay, (0, z['top']), (img_bgr.shape[1], z['bot']), color, -1)
            manifest.append({"idx": i, "y_top": z['top'], "y_bot": z['bot'], "state": z['state']})
        
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)
        
        # Add Search Limit Lines
        cv2.line(img_bgr, (0, SCAN_Y_START), (img_bgr.shape[1], SCAN_Y_START), (255, 100, 0), 2)
        cv2.line(img_bgr, (0, SCAN_Y_END), (img_bgr.shape[1], SCAN_Y_END), (255, 100, 0), 2)
        
        img_bgr = draw_pixel_ruler(img_bgr)
        img_bgr = draw_telemetry(img_bgr, intensities)
        
        cv2.putText(img_bgr, f"EPSILON F:{i} | BANNERS: {len(zones)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(OUT_DIR, f"epsilon_{i:05}.png"), img_bgr)

    pd.DataFrame(manifest).to_csv("sentinel_epsilon_manifest.csv", index=False)
    print(f"Sentinel Epsilon finished. Scan range: {SCAN_Y_START}-{SCAN_Y_END}")

if __name__ == "__main__":
    run_sentinel_epsilon()