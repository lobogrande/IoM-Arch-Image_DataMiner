import os

# project_config.py

# ==============================================================================
# STATIC BOSS FLOORS
# ==============================================================================
ASC_BOSS_DATA = {
    "asc1": {
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
    },
    "asc2": {
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
        80: {'tier': 'dirt3'}, 95: {'tier': 'com3'},
        98: {'tier': 'myth3'},
        99: {
        "tier": "mixed",
        "special": {
            0: "com4", 1: "rare3", 2: "epic3", 3: "leg3", 4: "myth3", 5: "div2",
            6: "com4",  7: "rare3",  8: "epic3",  9: "leg3",  10: "myth3", 11: "div2",
            12: "com4", 13: "rare3", 14: "epic3", 15: "leg3", 16: "myth3", 17: "div2",
            18: "com4", 19: "rare3", 20: "epic3", 21: "leg3", 22: "myth3", 23: "div2"
        }
        },
        110: {'tier': 'rare3'}, 125: {'tier': 'epic3'}, 135: {'tier': 'leg3'}, 
        140: {'tier': 'myth3'}, 149: {'tier': 'div3'}
    }
}

# ==============================================================================
# ORE APPEARANCE RANGES
# Format: 'block_id': (min_floor, max_floor)
# ==============================================================================
ASC_ORE_RESTRICTIONS = {
    "asc1": {
        'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
        'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
        'dirt3': (24, 999), 'com3': (30, 999), 'rare3': (36, 999), 'epic3': (42, 999), 'leg3': (45, 999), 'myth3': (50, 999), 'div3': (100, 999)
    },
    "asc2": {
        'dirt1': (1, 11), 'com1': (1, 17), 'rare1': (3, 25), 'epic1': (6, 29), 'leg1': (12, 31), 'myth1': (20, 34), 'div1': (50, 74),
        'dirt2': (12, 23), 'com2': (18, 28), 'rare2': (26, 35), 'epic2': (30, 41), 'leg2': (32, 44), 'myth2': (36, 49), 'div2': (75, 99),
        'dirt3': (24, 80), 'com3': (30, 95), 'rare3': (36, 110), 'epic3': (42, 125), 'leg3': (45, 135), 'myth3': (50, 140), 'div3': (100, 149),
        'dirt4': (81, 999), 'com4': (96, 999), 'rare4': (111, 999), 'epic4': (126, 999), 'leg4': (136, 999), 'myth4': (141, 999), 'div4': (150, 999)
    }
}

# ==============================================================================
# TEMPLATE MATCHING PENALTIES (BULLY SHIELD)
# Prevents high-entropy tiers from stealing matches from similar-looking blocks
# Based on template similarity analysis - tiers >95% similar need higher penalties
# ==============================================================================
ASC_BULLY_PENALTIES = {
    "asc1": {
        # Original ASC1 penalties (validated to work well)
        'rare1': 0.03, 'epic1': 0.04, 'leg1': 0.05,
        'rare2': 0.03, 'epic2': 0.04, 'leg2': 0.05,
        'dirt1': 0.04, 'dirt2': 0.04, 'dirt3': 0.04,
    },
    "asc2": {
        # Reduced leg penalties from 0.05 → 0.02 → 0.01 (same as dirt)
        # leg1 templates don't handle damage well; epic1 templates match damaged leg1 better
        # With 0.01 penalty: leg1 at 0.44 → 0.43, maximally competitive without eliminating penalty
        # Reduced dirt penalties from 0.04 to 0.01 - dirt blocks match well but lose to similar tiers
        # Reduced rare1/rare2 penalties from 0.03 to 0.00 - diagnostic shows rare1 raw scores consistently
        #   beat com1/com2, but even 0.01 penalty causes losses when gap is tiny (0.0019)
        # Added com1/com2 penalties of 0.01 - common blocks are easier to identify, shouldn't beat rare unfairly
        'rare1': 0.00, 'epic1': 0.04, 'leg1': 0.01,
        'rare2': 0.00, 'epic2': 0.04, 'leg2': 0.01,
        'com1': 0.01, 'com2': 0.01,
        'dirt1': 0.01, 'dirt2': 0.01, 'dirt3': 0.01, 'dirt4': 0.01,
        'div1': 0.06, 'div2': 0.06, 'div3': 0.06, 'div4': 0.06,
    }
}

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
def get_buffer_path(buffer_id=0, asc_level=None):
    """
    Returns absolute path to a specific capture buffer.
    
    Supports multiple naming patterns:
    - capture_buffer_0 (legacy, assumes asc1)
    - capture_buffer_asc1_0 (explicit asc1)
    - capture_buffer_asc2_0 (explicit asc2)
    
    Args:
        buffer_id: Buffer number (default 0)
        asc_level: Ascension level ("asc1" or "asc2"). If None, tries both patterns.
    
    Returns:
        Path to buffer directory (creates if doesn't exist)
    """
    base_dir = DATA_DIRS["RAW"]
    
    # If asc_level specified, use explicit naming
    if asc_level:
        buffer_name = f"capture_buffer_{asc_level}_{buffer_id}"
        path = os.path.join(base_dir, buffer_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    # Try to find existing buffer with either naming pattern
    # Priority: asc2 explicit > asc1 explicit > legacy
    for pattern in [f"capture_buffer_asc2_{buffer_id}", 
                    f"capture_buffer_asc1_{buffer_id}",
                    f"capture_buffer_{buffer_id}"]:
        path = os.path.join(base_dir, pattern)
        if os.path.exists(path):
            return path
    
    # If none exist, return legacy pattern (backward compatible)
    return os.path.join(base_dir, f"capture_buffer_{buffer_id}")

def get_ref_path(filename):
    """Returns absolute path to a file in the Reference/Ground Truth folder."""
    return os.path.join(DATA_DIRS["REF"], filename)

# Template/Digit Libraries
TEMPLATE_DIR = os.path.join(DATA_DIRS["REF"], "templates")
DIGIT_DIR = os.path.join(DATA_DIRS["REF"], "digits")

# --- 4. SHARED GAME CONSTANTS ---
# DEFAULT: Set to asc1 for backward compatibility
# Scripts should override this by calling set_ascension() based on their data
_CURRENT_ASCENSION = "asc1"

def set_ascension(asc_level):
    """
    Set the active ascension level (asc1 or asc2).
    This updates ORE_RESTRICTIONS, BOSS_DATA, and BULLY_PENALTIES to match the ascension.
    
    Usage in scripts:
        cfg.set_ascension("asc2")  # Before processing asc2 data
    """
    global _CURRENT_ASCENSION, ORE_RESTRICTIONS, BOSS_DATA, BULLY_PENALTIES
    if asc_level not in ASC_ORE_RESTRICTIONS:
        raise ValueError(f"Unknown ascension: {asc_level}. Must be 'asc1' or 'asc2'")
    _CURRENT_ASCENSION = asc_level
    ORE_RESTRICTIONS = ASC_ORE_RESTRICTIONS[asc_level]
    BOSS_DATA = ASC_BOSS_DATA[asc_level]
    BULLY_PENALTIES = ASC_BULLY_PENALTIES[asc_level]
    print(f"[Config] Set ascension to {asc_level.upper()}")
    print(f"[Config DEBUG] BOSS_DATA now has {len(BOSS_DATA)} floors: {list(BOSS_DATA.keys())[:5]}...{list(BOSS_DATA.keys())[-3:]}")
    print(f"[Config DEBUG] BULLY_PENALTIES loaded: {len(BULLY_PENALTIES)} tiers")

def get_ascension():
    """Get the currently active ascension level."""
    return _CURRENT_ASCENSION

# Initialize with default
ORE_RESTRICTIONS = ASC_ORE_RESTRICTIONS[_CURRENT_ASCENSION]
BOSS_DATA = ASC_BOSS_DATA[_CURRENT_ASCENSION]
BULLY_PENALTIES = ASC_BULLY_PENALTIES[_CURRENT_ASCENSION]


# Base stats for every block. 
# hp = Health, xp = Base XP, a = Armor, ft = Fragment Type (e.g., 0-6), fa = Fragment Amount
# project_config.py

# ==============================================================================
# ORE BASE STATS & SCALING BREAKPOINTS
# ft = Fragment Type (0: Dirt, 1: Com, 2: Rare, 3: Epic, 4: Leg, 5: Myth, 6: Div)
# fa = Fragment Amount
# ==============================================================================
BLOCK_BASE_STATS = {
    # ----------------------------------- TIER 1 -----------------------------------
    'dirt1': {'hp': 100,   'hp100': 200,   'hp150': 400,   'xp': 0.05, 'a': 0,   'a100': 0,    'a150': 0,    'ft': 0, 'fa': 0},
    'com1':  {'hp': 250,   'hp100': 500,   'hp150': 1000,  'xp': 0.15, 'a': 5,   'a100': 7.5,  'a150': 7.5,  'ft': 1, 'fa': 0.01},
    'rare1': {'hp': 550,   'hp100': 1100,  'hp150': 2200,  'xp': 0.35, 'a': 12,  'a100': 18,   'a150': 18,   'ft': 2, 'fa': 0.01},
    'epic1': {'hp': 1150,  'hp100': 2300,  'hp150': 4600,  'xp': 1.0,  'a': 25,  'a100': 38,   'a150': 38,   'ft': 3, 'fa': 0.01},
    'leg1':  {'hp': 1950,  'hp100': 3900,  'hp150': 7800,  'xp': 3.5,  'a': 50,  'a100': 75,   'a150': 75,   'ft': 4, 'fa': 0.01},
    'myth1': {'hp': 3500,  'hp100': 7000,  'hp150': 14000, 'xp': 7.5,  'a': 150, 'a100': 225,  'a150': 225,  'ft': 5, 'fa': 0.01},
    'div1':  {'hp': 25000, 'hp100': 50000, 'hp150': 100000,'xp': 20.0, 'a': 300, 'a100': 450,  'a150': 450,  'ft': 6, 'fa': 0.01},

    # ----------------------------------- TIER 2 -----------------------------------
    'dirt2': {'hp': 300,   'hp100': 600,   'hp150': 1200,  'xp': 0.15, 'a': 0,   'a100': 0,    'a150': 0,    'ft': 0, 'fa': 0},
    'com2':  {'hp': 750,   'hp100': 1500,  'hp150': 3000,  'xp': 0.45, 'a': 8,   'a100': 12,   'a150': 12,   'ft': 1, 'fa': 0.02},
    'rare2': {'hp': 1650,  'hp100': 3300,  'hp150': 6600,  'xp': 1.05, 'a': 20,  'a100': 30,   'a150': 30,   'ft': 2, 'fa': 0.02},
    'epic2': {'hp': 3450,  'hp100': 6900,  'hp150': 13800, 'xp': 3.0,  'a': 41,  'a100': 62,   'a150': 62,   'ft': 3, 'fa': 0.02},
    'leg2':  {'hp': 5850,  'hp100': 11700, 'hp150': 23400, 'xp': 10.5, 'a': 83,  'a100': 124,  'a150': 124,  'ft': 4, 'fa': 0.02},
    'myth2': {'hp': 10500, 'hp100': 21000, 'hp150': 42000, 'xp': 22.5, 'a': 248, 'a100': 371,  'a150': 371,  'ft': 5, 'fa': 0.02},
    'div2':  {'hp': 75000, 'hp100': 150000,'hp150': 300000,'xp': 60.0, 'a': 495, 'a100': 743,  'a150': 743,  'ft': 6, 'fa': 0.02},

    # ----------------------------------- TIER 3 -----------------------------------
    'dirt3': {'hp': 900,   'hp100': 1800,  'hp150': 3600,  'xp': 0.45, 'a': 0,   'a100': 0,    'a150': 0,    'ft': 0, 'fa': 0},
    'com3':  {'hp': 2250,  'hp100': 4500,  'hp150': 9000,  'xp': 1.35, 'a': 14,  'a100': 20,   'a150': 20,   'ft': 1, 'fa': 0.04},
    'rare3': {'hp': 4950,  'hp100': 9900,  'hp150': 19800, 'xp': 3.15, 'a': 33,  'a100': 49,   'a150': 49,   'ft': 2, 'fa': 0.04},
    'epic3': {'hp': 10350, 'hp100': 20700, 'hp150': 41400, 'xp': 9.0,  'a': 68,  'a100': 102,  'a150': 102,  'ft': 3, 'fa': 0.04},
    'leg3':  {'hp': 17550, 'hp100': 35100, 'hp150': 70200, 'xp': 31.5, 'a': 136, 'a100': 204,  'a150': 204,  'ft': 4, 'fa': 0.04},
    'myth3': {'hp': 31500, 'hp100': 63000, 'hp150': 126000,'xp': 67.5, 'a': 408, 'a100': 613,  'a150': 613,  'ft': 5, 'fa': 0.04},
    'div3':  {'hp': 225000,'hp100': 450000,'hp150': 900000,'xp': 180.0,'a': 817, 'a100': 1225, 'a150': 1225, 'ft': 6, 'fa': 0.04},

    # ----------------------------------- TIER 4 -----------------------------------
    'dirt4': {'hp': 2700,  'hp100': 5400,  'hp150': 10800, 'xp': 1.35, 'a': 0,   'a100': 0,    'a150': 0,    'ft': 0, 'fa': 0},
    'com4':  {'hp': 6750,  'hp100': 13500, 'hp150': 27000, 'xp': 4.05, 'a': 22,  'a100': 34,   'a150': 34,   'ft': 1, 'fa': 0.08},
    'rare4': {'hp': 14850, 'hp100': 29700, 'hp150': 59400, 'xp': 9.45, 'a': 54,  'a100': 81,   'a150': 81,   'ft': 2, 'fa': 0.08},
    'epic4': {'hp': 31050, 'hp100': 62100, 'hp150': 124200,'xp': 27.0, 'a': 112, 'a100': 168,  'a150': 168,  'ft': 3, 'fa': 0.08},
    'leg4':  {'hp': 52650, 'hp100': 105300,'hp150': 210600,'xp': 94.5, 'a': 225, 'a100': 337,  'a150': 337,  'ft': 4, 'fa': 0.08},
    'myth4': {'hp': 94500, 'hp100': 189000,'hp150': 378000,'xp': 202.5,'a': 674, 'a100': 1011, 'a150': 1011, 'ft': 5, 'fa': 0.08},
    'div4':  {'hp': 675000,'hp100': 1350000,'hp150': 2700000,'xp':540.0,'a': 1348,'a100': 2021, 'a150': 2021, 'ft': 6, 'fa': 0.08}
}

BASE_STAT_CAPS = {
    'Str': 50,
    'Agi': 50,
    'Per': 25,
    'Int': 25,
    'Luck': 25,
    'Div': 10,
    'Corr': 10
}

# ==============================================================================
# UPGRADE CAPS
# Defines the absolute maximum level a player can invest into an upgrade.
# ==============================================================================
INTERNAL_UPGRADE_CAPS = {
    3: 50,  4: 25,  5: 25, 8: 3,  9: 25,  10: 25, 11: 25, 12: 5,  13: 25, 
    14: 20, 15: 20, 16: 10, 17: 15, 18: 15, 19: 30, 20: 25, 21: 20, 
    22: 10, 23: 5,  24: 30, 25: 5,  26: 5,  27: 30,  28: 15, 29: 10, 
    30: 20, 31: 10, 32: 5,  33: 5,  34: 5, 35: 5,  36: 20, 37: 20, 
    38: 20, 39: 20, 40: 20, 41: 1,  42: 1,  43: 1,  44: 1,  45: 1,  
    46: 30, 47: 1,  48: 5,  49: 5,  50: 25, 51: 5,  52: 10, 53: 40, 
    54: 50, 55: 10
}

# ==============================================================================
# EXTERNAL UPGRADE UI GROUPS
# Maps unified UI controls (like a single checkbox) to multiple engine rows.
# ==============================================================================
EXTERNAL_UI_GROUPS =[
    {"id": "hestia", "name": "Hestia Idol", "rows": [4], "ui_type": "number", "max": 9999, "img": "4_hestia.png"},
    {"id": "axolotl", "name": "Axolotl Skin", "rows": [5], "ui_type": "pet", "max": 11, "img": "5_axolotl.png"},
    {"id": "dino", "name": "Dino Skin", "rows":[6, 7], "ui_type": "pet", "max": 11, "img": "6_7_dino.png"},
    {"id": "geoduck", "name": "Geoduck Tribute", "rows": [8], "ui_type": "number", "max": 9999, "img": "8_geoduck.png"},
    {"id": "avada", "name": "Avada Keda- Skill", "rows": [9, 10, 11], "ui_type": "skill", "imgs":["9_11_avada-keda_1.png", "9_11_avada-keda_2.png"]},
    {"id": "block", "name": "Block Bonker Skill", "rows":[12, 13, 14], "ui_type": "skill", "imgs":["12_14_block-bonker_1.png", "12_14_block-bonker_2.png"]},
    {"id": "arch_bundle", "name": "Archaeology Bundle", "rows": [15], "ui_type": "bundle"},
    {"id": "asc_bundle", "name": "Ascension Bundle", "rows": [16, 17, 18, 19], "ui_type": "bundle"},
    {"id": "arch_card", "name": "Arch Ability Card", "rows":[20], "ui_type": "card", "max": 4}
]