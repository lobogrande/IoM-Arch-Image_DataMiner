# ==============================================================================
# Script: app.py
# Layer 5: Streamlit Web UI
# Description: The visual frontend for the AI Arch Optimizer. Manages session
#              state, dynamic asset loading, and UI-to-Engine bridging.
# ==============================================================================

import streamlit as st
import json
import os
import sys

# --- PATH RESOLUTION ---
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json

# --- SESSION STATE INITIALIZATION ---
# This ensures the Player object survives Streamlit's constant page reruns
if 'player' not in st.session_state:
    st.session_state.player = Player()

p = st.session_state.player  # Quick reference variable

# --- UI CONFIGURATION ---
st.set_page_config(page_title="AI Arch Optimizer", layout="wide", page_icon="⛏️")

# ==========================================
# SIDEBAR: File Management & Global Settings
# ==========================================
with st.sidebar:
    st.header("📂 Player Data")
    st.write("Upload your save file to auto-fill the inputs.")
    
    uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
    
    # Process the uploaded file
    if uploaded_file is not None:
        # Streamlit holds files in memory. We write it to a temp file so our 
        # existing verify_player.py script can parse it with the Rosetta Stone!
        temp_path = os.path.join(ROOT_DIR, "temp_upload.json")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load it and immediately delete the temp file
        load_state_from_json(p, temp_path)
        os.remove(temp_path)
        st.success("Save file loaded!")
    
    st.divider()
    
    st.header("⚙️ Global Settings")
    # Tying the UI widgets directly to the Player object properties
    p.asc2_unlocked = st.checkbox("Ascension 2 Unlocked", value=p.asc2_unlocked)
    p.arch_level = st.number_input("Arch Level", min_value=1, value=p.arch_level)
    p.current_max_floor = st.number_input("Max Floor Reached", min_value=1, value=p.current_max_floor)
    
    st.divider()
    st.success("Engine Ready.")

# ==========================================
# MAIN WINDOW: Tabs
# ==========================================
st.title("⛏️ AI Arch Mining Optimizer")

tab_stats, tab_upgrades, tab_cards, tab_optimizer = st.tabs([
    "📊 Base Stats", 
    "⬆️ Upgrades", 
    "🃏 Cards", 
    "🚀 Run Optimizer"
])

# --- TAB 1: BASE STATS ---
with tab_stats:
    st.subheader("Base Stat Allocation")
    
    col1, col2, col3, col4 = st.columns(4)
    # Update Player object dynamically when these numbers change
    with col1:
        p.base_stats['Str'] = st.number_input("Strength (Str)", min_value=0, value=p.base_stats.get('Str', 0))
        p.base_stats['Agi'] = st.number_input("Agility (Agi)", min_value=0, value=p.base_stats.get('Agi', 0))
    with col2:
        p.base_stats['Per'] = st.number_input("Perception (Per)", min_value=0, value=p.base_stats.get('Per', 0))
        p.base_stats['Int'] = st.number_input("Intelligence (Int)", min_value=0, value=p.base_stats.get('Int', 0))
    with col3:
        p.base_stats['Luck'] = st.number_input("Luck (Luck)", min_value=0, value=p.base_stats.get('Luck', 0))
        p.base_stats['Div'] = st.number_input("Divinity (Div)", min_value=0, value=p.base_stats.get('Div', 0))
    with col4:
        if p.asc2_unlocked:
            p.base_stats['Corr'] = st.number_input("Corruption (Corr)", min_value=0, value=p.base_stats.get('Corr', 0))
        else:
            p.base_stats['Corr'] = 0
            st.info("Corruption is locked until Ascension 2.")

# --- TAB 2: UPGRADES ---
with tab_upgrades:
    st.subheader("Internal Upgrades")
    
    # Locked Asc2 Rows from your game design
    asc2_locked_rows =[17, 19, 34, 46, 52, 55]
    
    # Create a clean 3-column grid
    cols = st.columns(3)
    col_idx = 0
    
    for upg_id, upg_data in p.UPGRADE_DEF.items():
        name = upg_data[0]
        
        # Hide Asc2 upgrades if not unlocked
        if not p.asc2_unlocked and upg_id in asc2_locked_rows:
            continue
            
        current_col = cols[col_idx % 3]
        
        with current_col:
            # 1. Image Loading Logic
            img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "internal", f"{upg_id}.png")
            
            # Use an expander/container to keep it visually grouped
            with st.container(border=True):
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    # Fallback if image isn't captured yet
                    st.write(f"**[{upg_id}] {name}**")
                
                # 2. Input Logic
                current_lvl = p.upgrade_levels.get(upg_id, 0)
                # Key is required so Streamlit knows which widget is which
                new_lvl = st.number_input(f"Level##int_{upg_id}", min_value=0, value=current_lvl, label_visibility="collapsed")
                p.set_upgrade_level(upg_id, new_lvl)
                
        col_idx += 1

# --- TAB 3: CARDS ---
with tab_cards:
    st.subheader("Card Collection")
    st.write("Card UI coming soon...")

# --- TAB 4: OPTIMIZER ---
with tab_optimizer:
    st.subheader("Target Optimization")
    st.write("Optimizer hooks coming soon...")