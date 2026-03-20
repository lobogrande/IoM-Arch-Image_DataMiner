import cv2
import os

DIGITS_DIR = "digits"
HEADER_MAX_HEIGHT = 12 # Our current Y2-Y1

def audit_template_geometry():
    print(f"--- TEMPLATE GEOMETRY AUDIT ---")
    print(f"Target Height: {HEADER_MAX_HEIGHT}px (Header) | 16px (Dig Stage)")
    
    for f in sorted(os.listdir(DIGITS_DIR)):
        if not f.endswith('.png'): continue
        img = cv2.imread(os.path.join(DIGITS_DIR, f), 0)
        h, w = img.shape
        
        status = "OK" if h <= HEADER_MAX_HEIGHT else "TOO TALL FOR HEADER"
        print(f" {f:25} | Size: {w}x{h} | {status}")

if __name__ == "__main__":
    audit_template_geometry()