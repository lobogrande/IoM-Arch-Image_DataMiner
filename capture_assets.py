# ==============================================================================
# Script: capture_assets.py
# Description: A safe screen snipping tool. Features a 10ms polling loop to 
#              guarantee the floating box is always visible, and uses absolute 
#              global mouse tracking to perfectly bypass macOS Menu Bar offsets.
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
        
        # Force Tkinter to calculate window offsets (Menu Bar shift)
        self.root.update_idletasks()
        
        self.start_abs_x = None
        self.start_abs_y = None
        self.rect = None
        
        if self.preset_size:
            # PRESET MODE: Create box and start 10ms polling loop
            self.rect = self.canvas.create_rectangle(0, 0, 0, 0, outline='red', width=2)
            self.update_floating_box()
            self.canvas.bind("<ButtonPress-1>", self.on_preset_click)
        else:
            # FREEHAND MODE: Standard click and drag
            self.canvas.bind("<ButtonPress-1>", self.on_freehand_click)
            self.canvas.bind("<B1-Motion>", self.on_freehand_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_freehand_release)
        
        self.root.bind("<Escape>", self.on_cancel)
        self.canvas.bind("<Button-2>", self.on_cancel)
        self.canvas.bind("<Button-3>", self.on_cancel)

    # --- PRESET MODE LOGIC ---
    def update_floating_box(self):
        """10ms loop: Guarantees the box smoothly follows the cursor."""
        if self.preset_size and self.rect:
            w, h = self.preset_size
            
            # 1. Get absolute monitor coordinates
            abs_x = self.root.winfo_pointerx()
            abs_y = self.root.winfo_pointery()
            
            # 2. Subtract Tkinter's window offset so the Canvas draws it in the right visual spot
            cx = abs_x - self.root.winfo_rootx()
            cy = abs_y - self.root.winfo_rooty()
            
            self.canvas.coords(self.rect, cx - w//2, cy - h//2, cx + w//2, cy + h//2)
            
            # Rerun this function in 10 milliseconds
            self.root.after(10, self.update_floating_box)

    def on_preset_click(self, event):
        w, h = self.preset_size
        # Use raw absolute monitor coordinates for the actual screenshot
        abs_x = self.root.winfo_pointerx()
        abs_y = self.root.winfo_pointery()
        
        x = max(0, int(abs_x - w//2))
        y = max(0, int(abs_y - h//2))
        
        self.capture_coords = (x, y, int(w), int(h))
        self.root.quit()
        self.root.destroy()

    # --- FREEHAND MODE LOGIC ---
    def on_freehand_click(self, event):
        self.start_abs_x = self.root.winfo_pointerx()
        self.start_abs_y = self.root.winfo_pointery()
        
        cx = self.start_abs_x - self.root.winfo_rootx()
        cy = self.start_abs_y - self.root.winfo_rooty()
        
        self.rect = self.canvas.create_rectangle(cx, cy, cx, cy, outline='red', width=2)

    def on_freehand_drag(self, event):
        if self.rect:
            cur_abs_x = self.root.winfo_pointerx()
            cur_abs_y = self.root.winfo_pointery()
            
            cx1 = self.start_abs_x - self.root.winfo_rootx()
            cy1 = self.start_abs_y - self.root.winfo_rooty()
            cx2 = cur_abs_x - self.root.winfo_rootx()
            cy2 = cur_abs_y - self.root.winfo_rooty()
            
            self.canvas.coords(self.rect, cx1, cy1, cx2, cy2)

    def on_freehand_release(self, event):
        end_abs_x = self.root.winfo_pointerx()
        end_abs_y = self.root.winfo_pointery()
        
        x = max(0, int(min(self.start_abs_x, end_abs_x)))
        y = max(0, int(min(self.start_abs_y, end_abs_y)))
        w = int(abs(end_abs_x - self.start_abs_x))
        h = int(abs(end_abs_y - self.start_abs_y))
        
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
            last_size = (w, h)
            
            time.sleep(0.2) 
            os.system(f"screencapture -x -R {x},{y},{w},{h} '{app.filepath}'")
            print(f"✅ Saved perfectly aligned image to: {app.filepath}")

if __name__ == "__main__":
    main()