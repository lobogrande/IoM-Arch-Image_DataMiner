# ==============================================================================
# Script: capture_assets.py
# Description: A safe screen snipping tool that uses a transparent Tkinter 
#              overlay. Features a "Floating Preset Mode" to perfectly 
#              standardize crop sizes for uniform UI assets.
# ==============================================================================

import os
import time
import tkinter as tk

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')

class SnipOverlay:
    def __init__(self, root, filename, preset_size=None):
        self.root = root
        self.filename = filename
        self.filepath = os.path.join(ASSETS_DIR, filename)
        self.capture_coords = None  
        self.preset_size = preset_size # Tuple: (width, height)
        
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
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
        
        # --- BINDINGS BASED ON MODE ---
        if self.preset_size:
            # Preset Mode: Floating box that captures on a single click
            self.canvas.bind("<Motion>", self.on_hover)
            self.canvas.bind("<ButtonPress-1>", self.on_preset_click)
        else:
            # Freehand Mode: Click and drag to define a new box
            self.canvas.bind("<ButtonPress-1>", self.on_click)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        self.root.bind("<Escape>", self.on_cancel)
        self.canvas.bind("<Button-2>", self.on_cancel)
        self.canvas.bind("<Button-3>", self.on_cancel)

    # --- PRESET MODE LOGIC ---
    def on_hover(self, event):
        """Moves the standardized red box to follow the cursor."""
        w, h = self.preset_size
        x1, y1 = event.x - w//2, event.y - h//2
        x2, y2 = event.x + w//2, event.y + h//2
        
        if self.rect:
            self.canvas.coords(self.rect, x1, y1, x2, y2)
        else:
            self.rect = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline='red', width=2
            )

    def on_preset_click(self, event):
        """Locks the box and captures instantly."""
        w, h = self.preset_size
        x1_root = event.x_root - w//2
        y1_root = event.y_root - h//2
        
        self.capture_coords = (x1_root, y1_root, w, h)
        self.root.quit()
        self.root.destroy()

    # --- FREEHAND MODE LOGIC ---
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
            self.capture_coords = (x, y, w, h)
        else:
            print(">> Box too small. Capture cancelled.")
            
        self.root.quit()
        self.root.destroy()

    def on_cancel(self, event):
        self.root.quit()
        self.root.destroy()
        print(">> Capture cancelled.")


def main():
    print("=== AI Arch Asset Capturer (Standardized Dimensions) ===")
    print("1. Type your filename (e.g. 'stats/str') and press Enter.")
    print("2. The script will remember your box size for the next image!")
    print("3. To draw a NEW size, prefix your filename with 'n ' (e.g. 'n upgrades/3').\n")
    
    last_size = None
    
    while True:
        print("-" * 50)
        
        if last_size:
            prompt = f">> Enter filename (Standard Box: {last_size[0]}x{last_size[1]}). Type 'n [name]' to redraw:\n>> "
        else:
            prompt = ">> Enter filename (or 'q' to quit):\n>> "
            
        raw_input = input(prompt).strip()
        
        if raw_input.lower() == 'q':
            print("Exiting tool. Happy UI building!")
            break
            
        if not raw_input:
            continue
            
        # Check if user wants to force a new freehand draw
        force_new = False
        if raw_input.lower().startswith('n '):
            force_new = True
            filename = raw_input[2:].strip()
        else:
            filename = raw_input
            
        if not filename.endswith('.png'):
            filename += '.png'
            
        use_size = None if force_new else last_size
            
        root = tk.Tk()
        app = SnipOverlay(root, filename, preset_size=use_size)
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
        
        root.mainloop() 
        
        if app.capture_coords:
            x, y, w, h = app.capture_coords
            
            # Save these dimensions for the next run!
            last_size = (w, h)
            
            time.sleep(0.2) 
            os.system(f"screencapture -x -R {x},{y},{w},{h} '{app.filepath}'")
            print(f"✅ Saved perfectly aligned image to: {app.filepath}")

if __name__ == "__main__":
    main()