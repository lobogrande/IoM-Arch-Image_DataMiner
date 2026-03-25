# ==============================================================================
# Script: capture_assets.py
# Description: A safe screen snipping tool that uses a transparent Tkinter 
#              overlay to prevent QuickTime/iPhone mirroring from registering 
#              drag events. Red-box artifact fixed via mainloop deferral.
# ==============================================================================

import os
import time
import tkinter as tk

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

class SnipOverlay:
    def __init__(self, root, filename):
        self.root = root
        self.filename = filename
        self.filepath = os.path.join(ASSETS_DIR, filename)
        self.capture_coords = None  # Store coordinates to capture AFTER window closes
        
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Borderless window to prevent Mac from moving us to a new desktop
        self.root.overrideredirect(True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        self.root.attributes('-alpha', 0.25)
        self.root.attributes('-topmost', True)
        self.root.config(bg='gray')
        self.root.config(cursor="crosshair")
        
        self.canvas = tk.Canvas(self.root, bg="gray", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.start_x = None
        self.start_y = None
        self.start_x_root = None
        self.start_y_root = None
        self.rect = None
        
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.root.bind("<Escape>", self.on_cancel)
        self.canvas.bind("<Button-2>", self.on_cancel)
        self.canvas.bind("<Button-3>", self.on_cancel)

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.start_x_root = event.x_root
        self.start_y_root = event.y_root
        
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline='red', width=2
        )

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x_root, end_y_root = event.x_root, event.y_root
        
        x = min(self.start_x_root, end_x_root)
        y = min(self.start_y_root, end_y_root)
        w = abs(end_x_root - self.start_x_root)
        h = abs(end_y_root - self.start_y_root)
        
        if w > 10 and h > 10:
            # Save the coordinates to capture later
            self.capture_coords = (x, y, w, h)
        else:
            print(">> Box too small. Capture cancelled.")
            
        # Shut down the UI completely
        self.root.quit()
        self.root.destroy()

    def on_cancel(self, event):
        self.root.quit()
        self.root.destroy()
        print(">> Capture cancelled.")

def main():
    print("=== AI Arch Safe Asset Capturer ===")
    print("1. Type your filename (e.g. 'stats/str') and press Enter.")
    print("2. Your screen will dim. Click and drag a red box over the target.")
    print("3. Release to capture. (Right-Click or ESC to cancel).\n")
    
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
            
        root = tk.Tk()
        app = SnipOverlay(root, filename)
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
        
        # This blocks until app.root.quit() is called
        root.mainloop() 
        
        # Now the UI is 100% gone. If we have coordinates, take the pristine screenshot!
        if app.capture_coords:
            x, y, w, h = app.capture_coords
            time.sleep(0.2) # Give macOS a split second to clear the desktop buffer
            os.system(f"screencapture -x -R {x},{y},{w},{h} '{app.filepath}'")
            print(f"✅ Saved perfectly aligned image to: {app.filepath}")

if __name__ == "__main__":
    main()