# ==============================================================================
# Script: app.py
# Layer 5: Streamlit Web UI
# Description: Features perfect CSS Flexbox centering for Text and Images using
#              a custom Base64 HTML injection engine.
# ==============================================================================

import streamlit as st
import json
import os
import sys
import math
import glob
import base64
from io import BytesIO
from PIL import Image

# ==============================================================================
# 🎨 UI TWEAK PANEL 🎨
# ==============================================================================
UI_INT_COL_RATIO = [1, 1, 1]  

UI_EXT_GRID_COLS = 5
UI_EXT_IMG_STD     = 100  
UI_EXT_IMG_CARD    = 80   
UI_EXT_SKILL_ICON  = 50   
UI_EXT_SKILL_TEXT  = 160  
# ==============================================================================

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json
import project_config as cfg

# --- AUTO-CLAMPING CALLBACK ---
def enforce_caps(key, min_val, max_val, item_name):
    val = st.session_state[key]
    if val > max_val:
        st.session_state[key] = int(max_val)
        st.toast(f"⚠️ **{item_name}** exceeds limit. Clamped to Max ({max_val}).")
    elif val < min_val:
        st.session_state[key] = int(min_val)
        st.toast(f"⚠️ **{item_name}** below limit. Clamped to Min ({min_val}).")

def update_external_group(group_id, rows):
    val = st.session_state[group_id]
    for r in rows:
        st.session_state.player.set_external_level(r, int(val))

# --- IMAGE CENTERING & COMPOSITING HELPERS ---
def render_centered_image(img_source, width):
    """Bypasses Streamlit's left-alignment by converting the image to Base64 and centering it via HTML."""
    if isinstance(img_source, str):
        # It's a file path
        with open(img_source, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
    else:
        # It's a PIL Image object (from our card compositing)
        buffered = BytesIO()
        img_source.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        
    html = f"""
    <div style="display: flex; justify-content: center; margin-bottom: 10px;">
        <img src="data:image/png;base64,{encoded}" width="{width}px">
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def composite_card(bg_path):
    core_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", "20_Misc_Arch_Ability_face.png")
    try:
        bg = Image.open(bg_path).convert("RGBA")
        fg = Image.open(core_path).convert("RGBA")
        offset_x = (bg.width - fg.width) // 2
        offset_y = (bg.height - fg.height) // 2
        composite = bg.copy()
        composite.paste(fg, (offset_x, offset_y), mask=fg)
        return composite
    except Exception as e:
        return None

def find_external_image(upg_id):
    pattern = os.path.join(ROOT_DIR, "assets", "upgrades", "external", f"{upg_id}_*.png")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    exact = os.path.join(ROOT_DIR, "assets", "upgrades", "external", f"{upg_id}.png")
    return exact if os.path.exists(exact) else None

# --- SESSION STATE INITIALIZATION ---
if 'player' not in st.session_state:
    st.session_state.player = Player()

p = st.session_state.player
st.set_page_config(page_title="AI Arch Optimizer", layout="wide", page_icon="⛏️")

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📂 Player Data")
    uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
    
    if uploaded_file is not None:
        temp_path = os.path.join(ROOT_DIR, "temp_upload.json")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_state_from_json(p, temp_path)
        os.remove(temp_path)
        
        for k in list(st.session_state.keys()):
            if k.startswith("upg_") or k.startswith("stat_") or k.startswith("ext_"):
                del st.session_state[k]
        st.success("Save file loaded!")
    
    st.divider()
    st.header("⚙️ Global Settings")
    p.asc2_unlocked = st.checkbox("Ascension 2 Unlocked", value=p.asc2_unlocked)
    p.arch_level = st.number_input("Arch Level", min_value=1, value=int(p.arch_level), step=1)
    p.current_max_floor = st.number_input("Max Floor Reached", min_value=1, value=int(p.current_max_floor), step=1)

# ==========================================
# MAIN WINDOW: Tabs
# ==========================================
st.title("⛏️ AI Arch Mining Optimizer")

cap_inc = int(p.u('H45'))
STAT_CAPS = {
    'Str': 50 + cap_inc, 'Agi': 50 + cap_inc,
    'Per': 25 + cap_inc, 'Int': 25 + cap_inc, 'Luck': 25 + cap_inc,
    'Div': 10 + cap_inc, 'Corr': 10 + cap_inc
}

tab_stats, tab_upgrades, tab_cards, tab_optimizer = st.tabs([
    "📊 Base Stats", "⬆️ Upgrades", "🃏 Cards", "🚀 Run Optimizer"
])

with tab_stats:
    st.subheader("Base Stat Allocation")
    
    def render_stat(label, stat_key):
        max_val = int(STAT_CAPS[stat_key])
        current_val = int(p.base_stats.get(stat_key, 0))
        safe_val = min(max(current_val, 0), max_val)
        widget_key = f"stat_{stat_key}"
        
        if widget_key not in st.session_state:
            st.session_state[widget_key] = safe_val
        
        st.number_input(
            f"{label} (Max: {max_val})",
            key=widget_key, step=1, on_change=enforce_caps,
            args=(widget_key, 0, max_val, label)
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

with tab_upgrades:
    sub_internal, sub_external = st.tabs(["Internal Upgrades", "External Upgrades"])
    
    with sub_internal:
        asc2_locked_rows =[17, 19, 34, 46, 52, 55]
        active_upgrades =[]
        for upg_id, upg_data in p.UPGRADE_DEF.items():
            if not p.asc2_unlocked and upg_id in asc2_locked_rows:
                continue
            active_upgrades.append((upg_id, upg_data))
            
        spacer_left, center_col, spacer_right = st.columns(UI_INT_COL_RATIO)
        
        with center_col:
            for i, (upg_id, upg_data) in enumerate(active_upgrades):
                name = upg_data[0]
                max_lvl = int(cfg.INTERNAL_UPGRADE_CAPS.get(upg_id, 99))
                current_lvl = int(p.upgrade_levels.get(upg_id, 0))
                safe_val = min(max(current_lvl, 0), max_lvl)
                widget_key = f"upg_{upg_id}"
                
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = safe_val
                    
                img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "internal", f"{upg_id}.png")
                
                with st.container(border=True):
                    # Centered Title for Internal Upgrades
                    st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'><b>{name}</b><br><small>(Max: {max_lvl})</small></div>", unsafe_allow_html=True)
                    
                    if os.path.exists(img_path):
                        # Internal upgrades can still use container width since they are in a heavily constrained column, 
                        # but you can swap this for render_centered_image(img_path, 200) if you prefer!
                        st.image(img_path, use_container_width=True)
                    
                    st.number_input(
                        f"Level##int_{upg_id}", key=widget_key, step=1, 
                        on_change=enforce_caps, args=(widget_key, 0, max_lvl, name),
                        label_visibility="collapsed"
                    )
                    p.set_upgrade_level(upg_id, st.session_state[widget_key])

    with sub_external:
        cols_ext = st.columns(UI_EXT_GRID_COLS)
        
        for idx, group in enumerate(cfg.EXTERNAL_UI_GROUPS):
            widget_key = f"ext_{group['id']}"
            ui_type = group['ui_type']
            rows = group['rows']
            
            current_val = int(p.external_levels.get(rows[0], 0))
            if widget_key not in st.session_state:
                st.session_state[widget_key] = current_val

            with cols_ext[idx % UI_EXT_GRID_COLS]:
                with st.container(border=True):
                    
                    # Centered Title for External Upgrades
                    st.markdown(f"<div style='text-align: center; margin-bottom: 10px;'><b>{group['name']}</b></div>", unsafe_allow_html=True)
                    
                    # --- ASSET LOADING WITH CENTERED BASE64 HTML ---
                    if ui_type == "skill":
                        for img_name in group.get("imgs",[]):
                            img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "external", img_name)
                            if os.path.exists(img_path):
                                if "_1.png" in img_name:
                                    render_centered_image(img_path, UI_EXT_SKILL_ICON)
                                else:
                                    render_centered_image(img_path, UI_EXT_SKILL_TEXT)
                    elif "img" in group and group["img"]:
                        img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "external", group["img"])
                        if os.path.exists(img_path):
                            render_centered_image(img_path, UI_EXT_IMG_STD)
                    elif ui_type == "card":
                        tier = st.session_state[widget_key]
                        if tier > 0:
                            bg_path = os.path.join(ROOT_DIR, "assets", "cards", "backgrounds", f"{tier}.png")
                            comp_img = composite_card(bg_path)
                            if comp_img:
                                render_centered_image(comp_img, UI_EXT_IMG_CARD)
                            else:
                                st.markdown("<div style='text-align: center; color: gray;'>(Card Assets Missing)</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; color: gray;'>(Card Not Unlocked)</div>", unsafe_allow_html=True)

                    st.divider()

                    # --- WIDGET LOGIC ---
                    if ui_type in["number", "pet"]:
                        max_val = group.get("max", 999)
                        min_val = -1 if ui_type == "pet" else 0
                        
                        if ui_type == "pet" and st.session_state[widget_key] == -1:
                            st.markdown("<div style='text-align: center; color: gray;'><small>Status: Not Owned</small></div>", unsafe_allow_html=True)
                            
                        st.number_input(
                            f"Level##{group['id']}", min_value=min_val, max_value=max_val,
                            key=widget_key, step=1, on_change=update_external_group, args=(widget_key, rows),
                            label_visibility="collapsed"
                        )
                    
                    elif ui_type in["skill", "bundle"]:
                        is_checked = bool(st.session_state[widget_key])
                        def toggle_bool(k=widget_key, r=rows):
                            val = 1 if st.session_state[k] else 0
                            for row_id in r:
                                p.set_external_level(row_id, val)
                        st.checkbox("Unlocked", value=is_checked, key=widget_key, on_change=toggle_bool)
                        
                    elif ui_type == "card":
                        max_val = group.get("max", 4)
                        st.number_input(
                            f"Tier##{group['id']}", min_value=0, max_value=max_val,
                            key=widget_key, step=1, on_change=update_external_group, args=(widget_key, rows),
                            label_visibility="collapsed"
                        )

with tab_cards:
    st.subheader("Card Collection")
    st.write("Card UI coming soon...")

with tab_optimizer:
    st.subheader("Target Optimization")
    st.write("Optimizer hooks coming soon...")