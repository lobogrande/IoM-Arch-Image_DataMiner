# ==============================================================================
# Script: capture_assets.py
# Description: A safe screen snipping tool that uses a transparent Tkinter 
#              overlay to prevent QuickTime/iPhone mirroring from registering 
#              drag events. Requires ZERO third-party pip installs.
# ==============================================================================

import os
import time
import tkinter as tk

# Ensure it saves to the assets folder in your root directory
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

class SnipOverlay:
    def __init__(self, root, filename):
        self.root = root
        self.filename = filename
        self.filepath = os.path.join(ASSETS_DIR, filename)
        
        # Ensure target folder exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Configure the "Glass Shield" window
        self.root.attributes('-alpha', 0.2) # 20% opacity (dims screen slightly)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.config(bg='black')
        self.root.config(cursor="crosshair")
        
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        
        # Bind mouse events to the transparent canvas
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Press Esc to cancel without saving
        self.root.bind("<Escape>", self.on_cancel)

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        # Draw the starting point of the red bounding box
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline='red', width=2
        )

    def on_drag(self, event):
        if self.rect:
            # Expand the red bounding box as the mouse moves
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        self.root.destroy() # Destroy the glass shield
        
        # Calculate absolute capture bounds
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        w = abs(end_x - self.start_x)
        h = abs(end_y - self.start_y)
        
        if w > 5 and h > 5:  # Ignore accidental single clicks
            # Wait for the UI overlay to fully disappear from the screen
            time.sleep(0.3)
            # -x disables sound, -R captures specific coordinates
            os.system(f"screencapture -x -R {x},{y},{w},{h} '{self.filepath}'")
            print(f"✅ Saved successfully to: {self.filepath}")
        else:
            print(">> Box too small. Capture cancelled.")

    def on_cancel(self, event):
        self.root.destroy()
        print(">> Capture cancelled (Esc pressed).")

def main():
    print("=== AI Arch Safe Asset Capturer ===")
    print("This tool uses a protective overlay to prevent your game window from dragging.")
    print("1. Type your filename (e.g. 'stats/str') and press Enter.")
    print("2. Your screen will dim. Click and drag a red box over the target.")
    print("3. Release to capture. Press ESC if you make a mistake.\n")
    
    while True:
        print("-" * 50)
        filename = input(">> Enter filename (or 'q' to quit): ").strip()
        
        if filename.lower() == 'q':
            print("Exiting tool. Happy UI building!")
            break
            
        if not filename:
            continue
            
        if not filename.endswith('.png'):
            filename += '.png'
            
        # Create and launch the transparent Tkinter overlay
        root = tk.Tk()
        app = SnipOverlay(root, filename)
        root.mainloop()

if __name__ == "__main__":
    main()