# ==============================================================================
# Script: capture_assets.py
# Description: A rapid-fire screen snipping tool specifically for macOS. 
#              Triggers the native crosshair, captures to clipboard, and saves
#              directly into the assets/ directory based on terminal input.
# ==============================================================================

import os
import sys
from PIL import ImageGrab

# Ensure it saves to the assets folder in your root directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

def main():
    print("=== AI Arch Asset Capturer (macOS Edition) ===")
    print(f"Saving assets to: {ASSETS_DIR}\n")
    print("Instructions:")
    print(" 1. Press [ENTER] to start snipping.")
    print(" 2. Click and drag a box on your screen.")
    print(" 3. Type the folder/filename (e.g. 'stats/str') and hit enter.")
    print(" 4. Type 'q' instead of [ENTER] to exit.\n")
    
    while True:
        print("-" * 50)
        cmd = input("Press [ENTER] to capture, or type 'q' to quit: ").strip().lower()
        
        if cmd == 'q':
            print("Exiting tool. Happy UI building!")
            break
            
        print(">> Draw your rectangle on the screen now...")
        
        # Trigger macOS native interactive capture (-i) and copy to clipboard (-c)
        # This pauses the Python script until you finish drawing the box
        os.system("screencapture -i -c")
        
        # Retrieve the newly captured image from the Mac clipboard
        img = ImageGrab.grabclipboard()
        
        if img is None:
            print(">> Capture cancelled (you pressed Esc) or no image found.")
            continue
            
        # Prompt for the filename
        filename = input(">> Enter filename (e.g., 'stats/agi' or 'upgrades/4'): ").strip()
        
        if not filename:
            print(">> No filename provided. Discarding capture.")
            continue
            
        if not filename.endswith('.png'):
            filename += '.png'
            
        filepath = os.path.join(ASSETS_DIR, filename)
        
        # Automatically create the subdirectories if they don't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the image as a transparent PNG
        img.save(filepath, 'PNG')
        print(f"✅ Saved successfully to: {filepath}")

if __name__ == "__main__":
    main()