import cv2
import numpy as np
import mss
import pytesseract
from pynput import mouse
from PIL import Image, ImageOps

# --- CONFIGURATION ---
# PSM 7: Single text line. Char Whitelist: Digits only.
TESS_CONFIG = '--psm 7 -c tessedit_char_whitelist=0123456789'

# Global storage for clicks
clicks = []

def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.left:
        clicks.append((int(x), int(y)))
        if len(clicks) == 1:
            print(f"Captured Top-Left: {clicks[0]}")
            print("Now, click the BOTTOM-RIGHT corner of the digits.")
        elif len(clicks) == 2:
            print(f"Captured Bottom-Right: {clicks[1]}")
            return False # Stop the listener

def run_ocr_test(tl, br):
    # Calculate dimensions
    w = br[0] - tl[0]
    h = br[1] - tl[1]
    
    if w <= 0 or h <= 0:
        print("Error: Invalid area selected. Please run again.")
        return

    # Define ROI for MSS
    # Note: MSS handles screen coordinates directly
    test_roi = {'top': tl[1], 'left': tl[0], 'width': w, 'height': h}
    
    with mss.mss() as sct:
        # Capture the area
        sct_img = sct.grab(test_roi)
        # Convert to PIL Image for processing
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # 1. Grayscale
        img_gray = ImageOps.grayscale(img)
        # 2. Invert (Better for white-on-dark game text)
        img_inv = ImageOps.invert(img_gray)
        # 3. Save for visual verification
        img_inv.save("stage_debug.png")
        
        # Run OCR
        result = pytesseract.image_to_string(img_inv, config=TESS_CONFIG).strip()
        
        print("\n" + "="*30)
        print("--- CALIBRATION RESULTS ---")
        print(f"HEADER_ROI = {{'top': {tl[1]}, 'left': {tl[0]}, 'width': {w}, 'height': {h}}}")
        print(f"OCR Detected: '{result}'")
        print("="*30)
        
        if result:
            print(f"\nSUCCESS: '{result}' matches! Copy the HEADER_ROI into your Master Logger.")
        else:
            print("\nFAILURE: No digits detected. Check 'stage_debug.png' to see the crop.")

print("--- Interactive Stage Calibrator ---")
print("1. Click the TOP-LEFT corner of the floor number digits.")

# Start the listener
with mouse.Listener(on_click=on_click) as listener:
    listener.join()

# Process the results
run_ocr_test(clicks[0], clicks[1])