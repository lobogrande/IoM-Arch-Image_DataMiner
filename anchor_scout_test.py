import cv2
import numpy as np
import os

# --- CONFIG ---
TARGET_RUN = "0"
BUFFER_ROOT = f"capture_buffer_{TARGET_RUN}"
OUTPUT_DIR = f"diagnostic_results/BlindnessTest_v68"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_v68_blindness_test():
    # 1. VERIFY TEMPLATE LOADING
    tpl_path = "templates/player_right.png"
    player_t = cv2.imread(tpl_path, 0)
    
    if player_t is None:
        print(f"!!! CRITICAL ERROR: Could not load {tpl_path} !!!")
        print("Check if the file is in the 'templates' folder and named exactly 'player_right.png'")
        return
    else:
        h, w = player_t.shape
        print(f" [OK] Template Loaded: {w}x{h} pixels.")

    buffer_files = sorted([f for f in os.listdir(BUFFER_ROOT) if f.endswith(('.png', '.jpg'))])
    print(f" [OK] Found {len(buffer_files)} frames to scan.")

    # Floor 1 Force
    cv2.imwrite(os.path.join(OUTPUT_DIR, "Test_00000_START.jpg"), cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[0])))

    for i in range(len(buffer_files) - 1):
        # Heartbeat every 250 frames so we know it hasn't hung
        if i % 250 == 0: print(f" [Scanning] Frame {i}...", end='\r')

        img_bgr = cv2.imread(os.path.join(BUFFER_ROOT, buffer_files[i+1]))
        if img_bgr is None: continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 2. THE NUCLEAR SCAN
        # Searching the ENTIRE SCREEN for any 50% match
        res = cv2.matchTemplate(img_gray, player_t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        # If it's even slightly similar, SAVE IT.
        if max_val > 0.50:
            # Draw a box where it thinks the player is
            cv2.rectangle(img_bgr, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
            cv2.putText(img_bgr, f"Score: {max_val:.3f}", (30, 50), 0, 0.7, (0, 0, 255), 2)
            
            filename = f"Match_{i+1:05}_Score_{int(max_val*100)}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img_bgr)
            
            # Print to console so you see it live
            print(f"\n [!] Possible Match at Frame {i+1}! Score: {max_val:.3f}")
            
            # Limit to 50 images to avoid flooding
            if len(os.listdir(OUTPUT_DIR)) > 50:
                print("\n[FINISH] Found 50 samples. Test complete.")
                return

if __name__ == "__main__":
    run_v68_blindness_test()