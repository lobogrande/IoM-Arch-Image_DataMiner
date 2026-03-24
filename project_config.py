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
def get_buffer_path(buffer_id=4):
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
            0: 'com3', 1: 'com3', 2: 'com3', 3: 'com3', 4: 'com3', 5: 'com3',
            6: 'com3', 7: 'com3', 8: 'myth1', 9: 'myth1', 10: 'com3', 11: 'com3',
            12: 'com3', 13: 'com3', 14: 'myth1', 15: 'myth1', 16: 'com3', 17: 'com3',
            18: 'com3', 19: 'com3', 20: 'com3', 21: 'com3', 22: 'com3', 23: 'com3'
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

# Base stats for every ore. 
# hp = Health, xp = Base XP, a = Armor, ft = Fragment Type (e.g., 0-6), fa = Fragment Amount
ORE_BASE_STATS = {
    'dirt1': {'hp': 100, 'xp': 0.05, 'a': 0, 'ft': 0, 'fa': 0},
    'dirt2': {'hp': 300, 'xp': 0.15, 'a': 0, 'ft': 0, 'fa': 0},
    'dirt3': {'hp': 900, 'xp': 0.45, 'a': 0, 'ft': 0, 'fa': 0},
    
    'com1':  {'hp': 250, 'xp': 0.15, 'a': 5, 'ft': 1, 'fa': 0.01},
    'com2':  {'hp': 750, 'xp': 0.45, 'a': 8, 'ft': 1, 'fa': 0.02},
    'com3':  {'hp': 2250, 'xp': 1.35, 'a': 14, 'ft': 1, 'fa': 0.04},
    
    'rare1': {'hp': 550, 'xp': 0.35, 'a': 12, 'ft': 2, 'fa': 0.01},
    'rare2': {'hp': 1650, 'xp': 1.05, 'a': 20, 'ft': 2, 'fa': 0.02},
    'rare3': {'hp': 4950, 'xp': 3.15, 'a': 33, 'ft': 2, 'fa': 0.04},
    
    'epic1': {'hp': 1150, 'xp': 1, 'a': 25, 'ft': 3, 'fa': 0.01},
    'epic2': {'hp': 3450, 'xp': 3, 'a': 41, 'ft': 3, 'fa': 0.02},
    'epic3': {'hp': 10350, 'xp': 9, 'a': 68, 'ft': 3, 'fa': 0.04},
    
    'leg1':  {'hp': 1950, 'xp': 3.5, 'a': 50, 'ft': 4, 'fa': 0.01},
    'leg2':  {'hp': 5850, 'xp': 10.5, 'a': 83, 'ft': 4, 'fa': 0.02},
    'leg3':  {'hp': 17550, 'xp': 31.5, 'a': 136, 'ft': 4, 'fa': 0.04},
    
    'myth1': {'hp': 3500, 'xp': 7.5, 'a': 150, 'ft': 5, 'fa': 0.01},
    'myth2': {'hp': 10500, 'xp': 22.5, 'a': 248, 'ft': 5, 'fa': 0.02},
    'myth3': {'hp': 31500, 'xp': 67.5, 'a': 408, 'ft': 5, 'fa': 0.04},
    
    'div1':  {'hp': 25000, 'xp': 20, 'a': 300, 'ft': 6, 'fa': 0.01},
    'div2':  {'hp': 75000, 'xp': 60, 'a': 495, 'ft': 6, 'fa': 0.02},
    'div3':  {'hp': 225000, 'xp': 180, 'a': 817, 'ft': 6, 'fa': 0.04}
}