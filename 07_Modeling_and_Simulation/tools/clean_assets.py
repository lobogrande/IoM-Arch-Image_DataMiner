# ==============================================================================
# Script: tools/clean_assets.py
# Description: Interactive batch image editor. Lets the user draw a box over 
#              unwanted text (like dynamic stats) on one standardized image, 
#              and applies a seamless background-color patch to ALL images in 
#              the target directory.
# ==============================================================================

import os
import sys
import glob
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

class BatchCleanerApp:
    def __init__(self, root, image_paths):
        self.root = root
        self.image_paths = image_paths
        self.current_img_path = image_paths[0]
        
        # Load the first image to use as the template
        self.pil_image = Image.open(self.current_img_path).convert('RGB')
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        
        self.root.title(f"Batch Cleaner - {os.path.basename(self.current_img_path)}")
        self.root.geometry(f"{self.pil_image.width + 40}x{self.pil_image.height + 80}")
        
        # Instructions
        tk.Label(root, text="Draw a box over the text you want to hide.\nPress ENTER to apply to ALL images. Press ESC to cancel.", pady=5).pack()
        
        # Canvas matching the exact image size
        self.canvas = tk.Canvas(root, width=self.pil_image.width, height=self.pil_image.height, cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        
        # State variables
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.crop_coords = None
        
        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Return>", self.apply_to_all)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1, y1 = min(self.start_x, event.x), min(self.start_y, event.y)
        x2, y2 = max(self.start_x, event.x), max(self.start_y, event.y)
        self.crop_coords = (x1, y1, x2, y2)

    def apply_to_all(self, event):
        if not self.crop_coords:
            print("Please draw a box first!")
            return
            
        x1, y1, x2, y2 = self.crop_coords
        
        # Sample the background color just above the box (y1 - 2)
        # This grabs the dark gray of the UI box flawlessly
        sample_x = x1 + 2
        sample_y = max(0, y1 - 2)
        bg_color = self.pil_image.getpixel((sample_x, sample_y))
        
        target_dir = os.path.dirname(self.current_img_path)
        output_dir = os.path.join(target_dir, "cleaned")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Applying patch (Color: {bg_color}) to {len(self.image_paths)} images...")
        
        for path in self.image_paths:
            img = Image.open(path).convert('RGB')
            draw = ImageDraw.Draw(img)
            # Draw a solid rectangle over the text using the sampled background color
            draw.rectangle([x1, y1, x2, y2], fill=bg_color)
            
            out_path = os.path.join(output_dir, os.path.basename(path))
            img.save(out_path)
            
        print(f"✅ All {len(self.image_paths)} images cleaned and saved to: {output_dir}")
        self.root.destroy()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/clean_assets.py <path_to_folder>")
        print("Example: python tools/clean_assets.py assets/upgrades/internal")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    search_pattern = os.path.join(folder_path, "*.png")
    images = glob.glob(search_pattern)
    
    if not images:
        print(f"No PNG images found in {folder_path}")
        sys.exit(1)
        
    root = tk.Tk()
    app = BatchCleanerApp(root, images)
    
    # Bring window to front on Mac
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
    
    root.mainloop()