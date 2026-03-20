import cv2
import numpy as np
import os

# --- TARGET COORDINATES (Verified Legacy) ---
# Y1, Y2, X1, X2
HEADER_ROI = (54, 74, 103, 138) 

def run_hud_autopsy_v2():
    buffer_path = "capture_buffer_0"
    output_dir = "diagnostic_results/HUD_Autopsy_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. TRACE BACK TO RAW FILES BY INDEX
    if not os.path.exists(buffer_path):
        print(f"Error: Folder '{buffer_path}' not found.")
        return

    # Get all images and sort them alphabetically to ensure index 63 is consistent
    all_files = sorted([f for f in os.listdir(buffer_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(all_files) < 69:
        print(f"Error: Buffer only contains {len(all_files)} files. Cannot reach index 68.")
        return

    # Map indices to actual paths
    path_a = os.path.join(buffer_path, all_files[63])
    path_b = os.path.join(buffer_path, all_files[68])
    
    print(f"Targeting Index 63: {all_files[63]}")
    print(f"Targeting Index 68: {all_files[68]}")

    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    
    # 2. EXTRACT RAW ROIs
    y1, y2, x1, x2 = HEADER_ROI
    roi_a = img_a[y1:y2, x1:x2]
    roi_b = img_b[y1:y2, x1:x2]
    
    # 3. PROCESS (Adaptive Otsu)
    gray_a = cv2.cvtColor(roi_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2GRAY)
    
    # We use adaptive thresholding to see exactly what the OCR engine sees
    _, thresh_a = cv2.threshold(gray_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_b = cv2.threshold(gray_b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. PIXEL COMPARISON
    diff = cv2.absdiff(thresh_a, thresh_b)
    diff_score = np.sum(diff) / 255
    
    # 5. FORENSIC VISUALIZATION
    # Zooming in 600% to see sub-pixel jitter/compression artifacts
    scale = 6
    zoom_a = cv2.resize(thresh_a, (None, None), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    zoom_b = cv2.resize(thresh_b, (None, None), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    zoom_diff = cv2.resize(diff, (None, None), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # Vertical stack for better mobile/small screen viewing
    sheet = np.vstack((zoom_a, zoom_b, zoom_diff))
    
    # Labels
    cv2.putText(sheet, f"INDEX 63: {all_files[63]}", (5, 15), 0, 0.4, (255), 1)
    cv2.putText(sheet, f"INDEX 68: {all_files[68]}", (5, zoom_a.shape[0] + 15), 0, 0.4, (255), 1)
    cv2.putText(sheet, f"DIFF SCORE: {int(diff_score)}", (5, zoom_a.shape[0]*2 + 15), 0, 0.4, (255), 1)

    out_path = os.path.join(output_dir, "Forensic_HUD_Compare.png")
    cv2.imwrite(out_path, sheet)
    
    print(f"\n[DIAGNOSTIC RESULT]")
    print(f"Difference Score: {diff_score} pixels")
    print(f"Comparison image saved to: {out_path}")
    
    if diff_score > 0:
        print(f"Note: There are {int(diff_score)} unique pixels between these two frames.")
        print("We need to adjust our identity threshold to be > this number to reject them.")

if __name__ == "__main__":
    run_hud_autopsy_v2()