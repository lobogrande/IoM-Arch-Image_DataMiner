import os

# --- 1. PROJECT ROOT CALCULATION ---
# This allows scripts in subfolders to find the root regardless of where they are run
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- 2. DATA DIRECTORY MAPPING ---
DATA_DIRS = {
    "RAW": os.path.join(PROJECT_ROOT, "Data_00_Raw_Captures"),
    "REF": os.path.join(PROJECT_ROOT, "Data_01_Reference"),
    "TRACKING": os.path.join(PROJECT_ROOT, "Data_02_Tracking_Archive"),
    "SURGICAL": os.path.join(PROJECT_ROOT, "Data_03_Surgical_Mining_Results"),
    "CALIB": os.path.join(PROJECT_ROOT, "Data_04_Calibration_Vault"),
    "INVEST": os.path.join(PROJECT_ROOT, "Data_05_Investigation_Archives")
}

# --- 3. PATH HELPERS ---
def get_buffer_path(buffer_id=0):
    """Returns absolute path to a specific capture buffer."""
    return os.path.join(DATA_DIRS["RAW"], f"capture_buffer_{buffer_id}")

def get_ref_path(filename):
    """Returns absolute path to a file in the Reference/Ground Truth folder."""
    return os.path.join(DATA_DIRS["REF"], filename)

# Template/Digit Libraries
TEMPLATE_DIR = os.path.join(DATA_DIRS["REF"], "templates")
DIGIT_DIR = os.path.join(DATA_DIRS["REF"], "digits")

# --- 4. SHARED GAME CONSTANTS ---
ORE_RESTRICTIONS = {
    'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
    'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
    'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
}

BOSS_DATA = {
    11: {'tier': 'dirt1'}, 17: {'tier': 'com1'}, 23: {'tier': 'dirt2'},
    25: {'tier': 'rare1'}, 29: {'tier': 'epic1'}, 31: {'tier': 'leg1'},
    34: {
        'tier': 'mixed', 
        'special': {
            0: 'com2', 1: 'com2', 2: 'com2', 3: 'com2', 4: 'com2', 5: 'com2',
            6: 'com2', 7: 'com2', 8: 'myth1', 9: 'myth1', 10: 'com2', 11: 'com2',
            12: 'com2', 13: 'com2', 14: 'myth1', 15: 'myth1', 16: 'com2', 17: 'com2',
            18: 'com2', 19: 'com2', 20: 'com2', 21: 'com2', 22: 'com2', 23: 'com2'
        }
    },
    35: {'tier': 'rare2'}, 41: {'tier': 'epic2'}, 44: {'tier': 'leg2'},
    49: {
      "tier": "mixed",
      "special": {
        0: "dirt3", 1: "dirt3", 2: "dirt3", 3: "dirt3", 4: "dirt3", 5: "dirt3",
        6: "com3",  7: "com3",  8: "com3",  9: "com3",  10: "com3", 11: "com3",
        12: "rare3", 13: "rare3", 14: "rare3", 15: "rare3", 16: "rare3", 17: "rare3",
        18: "myth2", 19: "myth2", 20: "myth2", 21: "myth2", 22: "myth2", 23: "myth2"
      }
    },
    74: {
        'tier': 'mixed', 
        'special': {
            0: 'dirt3', 1: 'dirt3', 2: 'dirt3', 3: 'dirt3', 4: 'dirt3', 5: 'dirt3',
            6: 'dirt3', 7: 'dirt3', 8: 'dirt3', 9: 'dirt3', 10: 'dirt3', 11: 'dirt3',
            12: 'dirt3', 13: 'dirt3', 14: 'dirt3', 15: 'dirt3', 16: 'dirt3', 17: 'dirt3',
            18: 'dirt3', 19: 'dirt3', 20: 'div1', 21: 'div1', 22: 'dirt3', 23: 'dirt3'
        }
    },
    98: {'tier': 'myth3'},
    99: {
      "tier": "mixed",
      "special": {
        0: "com3", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2",
        6: "com3",  7: "rare3",  8: "epic3",  9: "leg3",  10: "myth3", 11: "div2",
        12: "com3", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2",
        18: "com3", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"
      }
    }
}