import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg

import pyautogui
from pynput import keyboard
import os

# --- CONFIGURE THESE FROM YOUR CALIBRATION RESULTS ---
GAME_X = 10  # Replace with your X
GAME_Y = 225  # Replace with your Y
# -----------------------------------------------------

print("Template Tool Active.")
print("Position your mouse over an Block or Shadow and press 'S' to save a 50x50 template.")
print("Press 'ESC' to stop.")

count = 0

def on_press(key):
    global count
    try:
        if key.char == 's':
            # Get current mouse position
            mouse_x, mouse_y = pyautogui.position()
            
            # Center the 50x50 box on the mouse
            snap_x = mouse_x - 25
            snap_y = mouse_y - 25
            
            # Capture the template
            img = pyautogui.screenshot(region=(snap_x, snap_y, 50, 50))
            
            filename = f"template_{count}.png"
            img.save(filename)
            print(f"Saved {filename} at screen pos ({mouse_x}, {mouse_y})")
            count += 1
            
    except AttributeError:
        if key == keyboard.Key.esc:
            print("Exiting tool...")
            return False

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()