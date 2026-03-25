# ==============================================================================
# Script: app.py
# Layer 5: Streamlit Web UI
# Description: Features seamless auto-clamping callbacks to instantly correct
#              out-of-bounds inputs without throwing UI errors.
# ==============================================================================

import streamlit as st
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json
import project_config as cfg

# --- AUTO-CLAMPING CALLBACK ---
def enforce_caps(key, min_val, max_val):
    """Instantly forces typed inputs into the allowed range without red errors."""
    val = st.session_state[key]
    if val > max_val:
        st.session_state[key] = max_val
    elif val < min_val:
        st.session_state[key] = min_val

# --- SESSION STATE INITIALIZATION ---
if 'player' not in st.session_state:
    st.session_state.player = Player()

p = st.session_state.player

st.set_page_config(page_title="AI Arch Optimizer", layout="wide", page_icon="⛏️")

# ==========================================
# SIDEBAR: File Management
# ==========================================
with st.sidebar:
    st.header("📂 Player Data")
    st.write("Upload your save file to auto-fill the inputs.")
    
    uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
    
    if uploaded_file is not None:
        temp_path = os.path.join(ROOT_DIR, "temp_upload.json")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        load_state_from_json(p, temp_path)
        os.remove(temp_path)
        
        # Flush existing widget keys so they resync to the new JSON file!
        for k in list(st.session_state.keys()):
            if k.startswith("upg_") or k.startswith("stat_"):
                del st.session_state[k]
                
        st.success("Save file loaded!")
    
    st.divider()
    
    st.header("⚙️ Global Settings")
    p.asc2_unlocked = st.checkbox("Ascension 2 Unlocked", value=p.asc2_unlocked)
    p.arch_level = st.number_input("Arch Level", min_value=1, value=p.arch_level)
    p.current_max_floor = st.number_input("Max Floor Reached", min_value=1, value=p.current_max_floor)

# ==========================================
# MAIN WINDOW: Tabs
# ==========================================
st.title("⛏️ AI Arch Mining Optimizer")

# Calculate dynamic Base Stat caps (Base + Upgrade #45)
cap_inc = int(p.u('H45'))
STAT_CAPS = {
    'Str': 50 + cap_inc, 'Agi': 50 + cap_inc,
    'Per': 25 + cap_inc, 'Int': 25 + cap_inc, 'Luck': 25 + cap_inc,
    'Div': 10 + cap_inc, 'Corr': 10 + cap_inc
}

tab_stats, tab_upgrades, tab_cards, tab_optimizer = st.tabs([
    "📊 Base Stats", "⬆️ Upgrades", "🃏 Cards", "🚀 Run Optimizer"
])

# --- TAB 1: BASE STATS ---
with tab_stats:
    st.subheader("Base Stat Allocation")
    
    def render_stat(label, stat_key):
        max_val = STAT_CAPS[stat_key]
        current_val = p.base_stats.get(stat_key, 0)
        safe_val = min(max(current_val, 0), max_val)
        widget_key = f"stat_{stat_key}"
        
        st.number_input(
            f"{label} (Max: {max_val})",
            value=safe_val,
            key=widget_key,
            on_change=enforce_caps,
            args=(widget_key, 0, max_val)
        )
        p.base_stats[stat_key] = st.session_state[widget_key]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat("Strength", 'Str')
        render_stat("Agility", 'Agi')
    with col2:
        render_stat("Perception", 'Per')
        render_stat("Intelligence", 'Int')
    with col3:
        render_stat("Luck", 'Luck')
        render_stat("Divinity", 'Div')
    with col4:
        if p.asc2_unlocked:
            render_stat("Corruption", 'Corr')
        else:
            p.base_stats['Corr'] = 0
            st.info("Corruption is locked until Ascension 2.")

# --- TAB 2: UPGRADES ---
with tab_upgrades:
    st.subheader("Internal Upgrades")
    
    asc2_locked_rows =[17, 19, 34, 46, 52, 55]
    cols = st.columns(3)
    col_idx = 0
    
    for upg_id, upg_data in p.UPGRADE_DEF.items():
        if not p.asc2_unlocked and upg_id in asc2_locked_rows:
            continue
            
        name = upg_data[0]
        max_lvl = cfg.INTERNAL_UPGRADE_CAPS.get(upg_id, 99)
        current_lvl = p.upgrade_levels.get(upg_id, 0)
        safe_val = min(max(current_lvl, 0), max_lvl)
        widget_key = f"upg_{upg_id}"
        
        current_col = cols[col_idx % 3]
        with current_col:
            img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "internal", f"{upg_id}.png")
            
            with st.container(border=True):
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                
                st.markdown(f"**[{upg_id}] {name}** (Max: {max_lvl})")
                
                # Render without native bounds to avoid Streamlit errors
                st.number_input(
                    f"Level##{upg_id}", 
                    value=safe_val, 
                    key=widget_key,
                    on_change=enforce_caps,
                    args=(widget_key, 0, max_lvl),
                    label_visibility="collapsed"
                )
                p.set_upgrade_level(upg_id, st.session_state[widget_key])
                
        col_idx += 1

# --- TAB 3: CARDS ---
with tab_cards:
    st.subheader("Card Collection")
    st.write("Card UI coming soon...")

# --- TAB 4: OPTIMIZER ---
with tab_optimizer:
    st.subheader("Target Optimization")
    st.write("Optimizer hooks coming soon...")