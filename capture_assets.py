# ==============================================================================
# Script: capture_assets.py
# Description: A safe screen snipping tool that uses a transparent Tkinter 
#              overlay to prevent QuickTime/iPhone mirroring from registering 
#              drag events. Includes perfect Retina Display scaling math.
# ==============================================================================

import os
import time
import tkinter as tk
from PIL import ImageGrab

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

class SnipOverlay:
    def __init__(self, root, filename):
        self.root = root
        self.filename = filename
        self.filepath = os.path.join(ASSETS_DIR, filename)
        
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        self.root.overrideredirect(True)
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
        
        self.root.attributes('-alpha', 0.25)
        self.root.attributes('-topmost', True)
        self.root.config(bg='gray')
        self.root.config(cursor="crosshair")
        
        self.canvas = tk.Canvas(self.root, bg="gray", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.start_x = None
        self.start_y = None
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
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, 
            outline='red', width=3
        )

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        end_x, end_y = event.x, event.y
        
        # Calculate logical bounds
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        w_logical = x2 - x1
        h_logical = y2 - y1
        
        # Destroy the glass shield so it isn't in the screenshot
        self.root.destroy()
        
        if w_logical > 10 and h_logical > 10:
            # Wait exactly 0.3 seconds for the window to completely fade from macOS
            time.sleep(0.3)
            
            try:
                # Grab the full physical monitor
                full_img = ImageGrab.grab(all_screens=False)
                
                # Calculate the exact Retina Scaling Factor
                scale_x = full_img.width / self.screen_width
                scale_y = full_img.height / self.screen_height
                
                # Convert the logical bounds to physical pixels
                crop_box = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                )
                
                # Crop and save
                final_img = full_img.crop(crop_box)
                final_img.save(self.filepath, 'PNG')
                print(f"✅ Saved perfectly aligned image to: {self.filepath}")
                
            except Exception as e:
                print(f"❌ Error processing image: {e}")
        else:
            print(">> Box too small. Capture cancelled.")

    def on_cancel(self, event):
        self.root.destroy()
        print(">> Capture cancelled.")

def main():
    print("=== AI Arch Safe Asset Capturer (Retina Fixed) ===")
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
        root.mainloop()

if __name__ == "__main__":
    main()