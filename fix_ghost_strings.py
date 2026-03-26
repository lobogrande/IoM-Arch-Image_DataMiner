import os
import re

TARGET_FOLDERS = [
    '01_Calibration_and_HUD', '02_Sentinel_Banner_Tracking', 
    '03_MultiStage_Worker_Pipeline', '04_Forensics_and_Profiling', 
    '05_Template_and_Signature_Library'
]

def fix_ghost_strings(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex looks for "capture_buffer_X/frame_..." and captures the buffer number and filename
    # It replaces it with os.path.join(cfg.get_buffer_path(X), "frame_...")
    pattern = r'["\']capture_buffer_(\d+)/(frame_.*?\.png)["\']'
    replacement = r'os.path.join(cfg.get_buffer_path(\1), "\2")'
    
    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for folder in TARGET_FOLDERS:
    if not os.path.exists(folder): continue
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                if fix_ghost_strings(os.path.join(root, file)):
                    print(f"[Fixed Ghost String] {file}")