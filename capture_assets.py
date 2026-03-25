# ==============================================================================
# Script: capture_assets.py
# Description: A safe screen snipping tool that uses a transparent Tkinter 
#              overlay to prevent QuickTime/iPhone mirroring from registering 
#              drag events. macOS Fullscreen Space Bug Fixed.
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
        
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # --- MAC SPECIFIC FIX ---
        # Instead of Official Fullscreen (which creates a new black desktop space),
        # we strip the window borders and size it to the monitor manually.
        self.root.overrideredirect(True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Configure the "Glass Shield"
        self.root.attributes('-alpha', 0.25) # 25% opacity
        self.root.attributes('-topmost', True)
        self.root.config(bg='gray')
        self.root.config(cursor="crosshair")
        
        self.canvas = tk.Canvas(self.root, bg="gray", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.start_x = None
        self.start_y = None
        self.rect = None
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Press Esc to cancel without saving
        self.root.bind("<Escape>", self.on_cancel)
        # Fallback binding for Mac if Esc doesn't trigger on borderless windows
        self.canvas.bind("<Button-2>", self.on_cancel) # Right click to cancel
        self.canvas.bind("<Button-3>", self.on_cancel)

    def on_click(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline='red', width=3
        )

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        self.root.destroy()
        
        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        w = abs(end_x - self.start_x)
        h = abs(end_y - self.start_y)
        
        if w > 10 and h > 10:
            time.sleep(0.3)
            os.system(f"screencapture -x -R {x},{y},{w},{h} '{self.filepath}'")
            print(f"✅ Saved successfully to: {self.filepath}")
        else:
            print(">> Box too small. Capture cancelled.")

    def on_cancel(self, event):
        self.root.destroy()
        print(">> Capture cancelled.")

def main():
    print("=== AI Arch Safe Asset Capturer ===")
    print("This tool uses a protective overlay to prevent your game window from dragging.")
    print("1. Type your filename (e.g. 'stats/str') and press Enter.")
    print("2. Your screen will dim. Click and drag a red box over the target.")
    print("3. Release to capture. (Right-Click or ESC to cancel if you make a mistake).\n")
    
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
        
        # Force the borderless window to the front on Mac
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
        
        root.mainloop()

if __name__ == "__main__":
    main()