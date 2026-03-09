import cv2
import numpy as np
import os
import json

# --- TARGET CONFIG ---
TARGET_RUN = "0"
TARGET_FLOOR = 49
INPUT_ROOT = "Unified_Consensus_Inputs"

# --- INITIAL LOAD ---
def get_target_image():
    run_path = os.path.join(INPUT_ROOT, f"Run_{TARGET_RUN}")
    json_path = os.path.join(run_path, "final_sequence.json")
    with open(json_path, 'r') as f:
        sequence = {e['floor']: e for e in json.load(f)}
    img_name = f"F{TARGET_FLOOR}_{sequence[TARGET_FLOOR]['frame']}"
    return cv2.imread(os.path.join(run_path, img_name))

def run_interactive_tuner():
    img_base = get_target_image()
    if img_base is None: return

    # Current "Best Guesses" from the project
    curr_x, curr_y = 75, 261
    step_x, step_y = 59.1, 59.1
    
    print("\n--- INTERACTIVE TUNER LOADED ---")
    print("WASD: Move Grid | ARROWS: Adjust Spacing | SPACE: Save & Exit")

    while True:
        temp_img = img_base.copy()
        
        # Draw the 24 slots
        for row in range(4):
            for col in range(6):
                # Calculate center
                cx = int(curr_x + (col * step_x))
                cy = int(curr_y + (row * step_y))
                
                # Draw Box (48x48)
                cv2.rectangle(temp_img, (cx-24, cy-24), (cx+24, cy+24), (0, 255, 0), 1)
                # Draw Center Crosshair
                cv2.line(temp_img, (cx-5, cy), (cx+5, cy), (0, 255, 0), 1)
                cv2.line(temp_img, (cx, cy-5), (cx, cy+5), (0, 255, 0), 1)

        # Status Overlay
        cv2.putText(temp_img, f"X:{curr_x} Y:{curr_y} | StepX:{step_x:.2f} StepY:{step_y:.2f}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Interactive Archaeology Tuner", temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        # WASD - Movement
        if key == ord('w'): curr_y -= 1
        elif key == ord('s'): curr_y += 1
        elif key == ord('a'): curr_x -= 1
        elif key == ord('d'): curr_x += 1
        
        # ARROWS - Step Adjustment (Use Shift or Alt if needed for smaller steps)
        elif key == 0 or key == 82: step_y -= 0.05 # Up Arrow
        elif key == 1 or key == 81: step_x -= 0.05 # Left Arrow
        elif key == 2 or key == 83: step_y += 0.05 # Down Arrow
        elif key == 3 or key == 84: step_x += 0.05 # Right Arrow
        
        elif key == 32: # SPACE to Save
            print(f"\nFINAL SETTINGS SAVED:")
            print(f"SLOT1_CENTER = ({curr_x}, {curr_y})")
            print(f"STEP_X, STEP_Y = {step_x:.3f}, {step_y:.3f}")
            break
        elif key == 27: # ESC to Quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_interactive_tuner()