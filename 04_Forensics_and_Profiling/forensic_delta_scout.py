import cv2
import numpy as np
import os

# --- TARGET GAP CONFIG ---
# Let's scout the 25->27 gap first (around Floor 25 in your v16 results)
# Replace these with the approximate frame indices from your v16 JSON
START_FRAME = 17482 
END_FRAME = 19383
BUFFER_ROOT = "capture_buffer_0"

def get_existence_vector(img_gray, bg_templates):
    vector = []
    for slot in range(24):
        row, col = divmod(slot, 6)
        cx, cy = int(74+(col*59.1)), int(261+(row*59.1))
        roi = img_gray[cy-8:cy+8, cx-8:cx+8]
        diff = min([np.sum(cv2.absdiff(roi, bg[16:32, 16:32])) / 256 for bg in bg_templates])
        vector.append(1 if diff > 8.5 else 0)
    return vector

def run_gap_forensics():
    bg_t = [cv2.resize(cv2.imread(f"templates/{f}", 0), (48, 48)) for f in os.listdir("templates") if f.startswith("background")]
    files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    
    print(f"--- Scouting Gap: {START_FRAME} to {END_FRAME} ---")
    for i in range(START_FRAME, END_FRAME):
        img_n = cv2.imread(os.path.join(BUFFER_ROOT, files[i]), 0)
        img_n1 = cv2.imread(os.path.join(BUFFER_ROOT, files[i+1]), 0)
        
        vec_n = get_existence_vector(img_n, bg_t)
        vec_n1 = get_existence_vector(img_n1, bg_t)
        spawn_slots = [j for j in range(24) if vec_n[j] == 0 and vec_n1[j] == 1]
        
        if len(spawn_slots) > 0:
            rows = set([divmod(s, 6)[0] for s in spawn_slots])
            print(f"Frame {i}: Spawns={len(spawn_slots)} | Rows={len(rows)} | DNA_Sum={sum(vec_n1)}")

if __name__ == "__main__":
    run_gap_forensics()