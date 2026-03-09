import cv2
import numpy as np
import os
import json
import sys
import re

# --- 1. MASTER BOSS DATA (UNABRIDGED) ---
# (Keeping full dictionary to ensure healer knows if a gap contains a boss floor)
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

DATASETS = ["0", "1", "2", "3", "4"]
BASE_HEAL_DIR = "Pass2_Evidence"

def run_worker2_healer():
    if not os.path.exists(BASE_HEAL_DIR): os.makedirs(BASE_HEAL_DIR)

    for ds_id in DATASETS:
        json_file = f"milestones_run_{ds_id}.json"
        buffer_path = f"capture_buffer_{ds_id}"
        if not os.path.exists(json_file) or not os.path.isdir(buffer_path): continue
        
        with open(json_file, 'r') as f:
            anchors = json.load(f)
            
        heal_path = os.path.join(BASE_HEAL_DIR, f"Run_{ds_id}")
        if not os.path.exists(heal_path): os.makedirs(heal_path)
        
        frames = sorted([f for f in os.listdir(buffer_path) if f.endswith(('.png', '.jpg'))])
        
        print(f"\n--- HEALING RUN {ds_id} ---")
        healed_milestones = []
        
        # 1. Identify Gaps between anchors
        for i in range(len(anchors) - 1):
            start_floor = anchors[i]['floor']
            end_floor = anchors[i+1]['floor']
            start_idx = anchors[i]['idx']
            end_idx = anchors[i+1]['idx']
            
            if (end_floor - start_floor) > 1:
                print(f" Gap Detected: Floor {start_floor} to {end_floor} (Frames {start_idx}-{end_idx})")
                # TODO: Implement the cascading scan pass here
                
        # 2. Find Dataset Endpoint
        last_anchor = anchors[-1]
        print(f" Final Anchor: Floor {last_anchor['floor']} at Frame {last_anchor['idx']}")
        # TODO: Scan remaining frames to find if floors exist after last anchor

if __name__ == "__main__":
    run_worker2_healer()