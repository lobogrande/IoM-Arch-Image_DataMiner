import os
import re

TARGET_FOLDERS = [
    '01_Calibration_and_HUD', '02_Sentinel_Banner_Tracking', 
    '03_MultiStage_Worker_Pipeline', '04_Forensics_and_Profiling', 
    '05_Template_and_Signature_Library', '06_Audit_and_Reporting','Infrastructure'
]

IMPORT_BLOCK = """import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_config as cfg
"""

def migrate_script(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    content = "".join(lines)
    
    # 1. Inject the Config Import if missing
    if "import project_config" not in content:
        content = IMPORT_BLOCK + "\n" + content

    # 2. Path Substitutions
    content = content.replace('"capture_buffer_0"', 'cfg.get_buffer_path(0)')
    content = content.replace("'capture_buffer_0'", 'cfg.get_buffer_path(0)')
    content = content.replace('"templates"', 'cfg.TEMPLATE_DIR')
    content = content.replace('"digits"', 'cfg.DIGIT_DIR')
    content = content.replace('"Final_FloorMap_v16.json"', 'cfg.get_ref_path("Final_FloorMap_v16.json")')

    # 3. Strip local BOSS_DATA declaration (Regex for multiline dict)
    # This looks for BOSS_DATA = { ... } and removes it
    content = re.sub(r'BOSS_DATA = \{.*?\n\}', '# BOSS_DATA moved to project_config', content, flags=re.DOTALL)
    content = re.sub(r'ORE_RESTRICTIONS = \{.*?\n\}', '# ORE_RESTRICTIONS moved to project_config', content, flags=re.DOTALL)

    # 4. Global variable replacement
    # Ensure code uses cfg.BOSS_DATA instead of just BOSS_DATA
    content = content.replace(' BOSS_DATA', ' cfg.BOSS_DATA')
    content = content.replace(' ORE_RESTRICTIONS', ' cfg.ORE_RESTRICTIONS')

    with open(file_path, 'w') as f:
        f.write(content)

def run_migration():
    for folder in TARGET_FOLDERS:
        if not os.path.exists(folder): continue
        print(f"Refactoring folder: {folder}...")
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".py") and file != "code_migrator.py":
                    print(f"  [Processing] {file}")
                    migrate_script(os.path.join(root, file))

if __name__ == "__main__":
    run_migration()