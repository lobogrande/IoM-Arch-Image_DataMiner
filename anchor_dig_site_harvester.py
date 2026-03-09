import cv2
import os

# Coordinates for the words "Dig Stage:" (Left of the number)
# X=161 to X=250 covers the text area
ANCHOR_ROI = (229, 248, 163, 253) 
DATASET_DIR = "capture_buffer_0"
TARGET_FRAME = "frame_20260306_231745_817144.png"

def harvest_text_anchor():
    img_path = os.path.join(DATASET_DIR, TARGET_FRAME)
    if not os.path.exists(img_path):
        print("Target frame not found."); return

    img = cv2.imread(img_path, 0)
    roi = img[ANCHOR_ROI[0]:ANCHOR_ROI[1], ANCHOR_ROI[2]:ANCHOR_ROI[3]]
    
    # We use a slightly lower threshold to ensure the letters connect
    _, bin_anchor = cv2.threshold(roi, 165, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite("dig_stage_anchor.png", bin_anchor)
    print("Saved 'dig_stage_anchor.png'. Verify that the words are legible.")

if __name__ == "__main__":
    harvest_text_anchor()