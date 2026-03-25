# ==============================================================================
# Script: app.py
# Layer 5: Streamlit Web UI
# Description: Features perfect CSS Flexbox centering for Text and Images using
#              a custom Base64 HTML injection engine. Includes global Stat Point 
#              budget tracking and auto-clamping.
# ==============================================================================

import streamlit as st
import json
import os
import sys
import math
import glob
import base64
import time
import multiprocessing as mp
import plotly.express as px
from io import BytesIO
from PIL import Image
import pandas as pd

# ==============================================================================
# 🎨 UI TWEAK PANEL 🎨
# Adjust these numbers, hit Save, and watch your browser instantly update!
# ==============================================================================

# --- BASE STATS ---
# Width of the stat icons
UI_STAT_IMG_WIDTH = 250

# --- INTERNAL UPGRADES ---
# The layout ratio for the single-column feed:[Left_Spacer, Center_Feed, Right_Spacer]
# To shrink the center box: Increase the outer numbers (e.g.,[2, 2, 2] or [1, 1, 1])
# To widen the center box: Increase the middle number (e.g.,[1, 3, 1])
UI_INT_COL_RATIO =[1, 1, 1]  

# --- EXTERNAL UPGRADES ---
# The layout ratio:[Left_Spacer, Content_Cols..., Right_Spacer]
# To get 3 centered columns with buffers:[1, 2, 2, 2, 1]
# To get 4 centered columns with buffers:[1, 2, 2, 2, 2, 1]
# The script always leaves the first and last array values completely empty!
UI_EXT_COL_RATIO =[1, 2, 2, 2, 1]

# Image Pixel Widths for External Upgrades
UI_EXT_IMG_STD     = 120  # Size of standard icons (Hestia, Geoduck, Dino)
UI_EXT_IMG_CARD    = 80   # Size of the composited Card
UI_EXT_SKILL_ICON  = 50   # Size of the Skill Icon (files ending in _1.png)
UI_EXT_SKILL_TEXT  = 250  # Size of the Skill Description (files ending in _2.png)

# Card Core Alignment
# Negative numbers move the core image UP. Positive numbers move it DOWN.
UI_EXT_CARD_CORE_Y_OFFSET = -4 

# --- ORE CARDS ---
# Width of the generated cards in the 4x7 grid
UI_ORE_CARD_WIDTH = 100
# Y-Offset specifically for Ore Card cores
UI_ORE_CARD_Y_OFFSET = -4

# Width of the ore icons inside the Ore Stats DataFrame table
UI_ORE_TABLE_IMG_WIDTH = 40
# ==============================================================================
# ==============================================================================

# --- PATH RESOLUTION ---
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

from core.player import Player
from core.ore import Ore
from tools.verify_player import load_state_from_json, save_state_to_json
import project_config as cfg
from optimizers.parallel_worker import run_optimization_phase, benchmark_hardware, get_eta_profiles

# --- AUTO-CLAMPING CALLBACKS ---
def enforce_caps(key, min_val, max_val, item_name):
    """Standard clamping for independent limits (e.g. Upgrade Levels)."""
    if key not in st.session_state: return 
    val = st.session_state[key]
    if val > max_val:
        st.session_state[key] = int(max_val)
        st.toast(f"⚠️ **{item_name}** exceeds limit. Clamped to Max ({max_val}).")
    elif val < min_val:
        st.session_state[key] = int(min_val)
        st.toast(f"⚠️ **{item_name}** below limit. Clamped to Min ({min_val}).")

def enforce_stat_caps(widget_key, stat_key, min_val, max_val, item_name):
    """Specialized clamping for Base Stats that also checks the Global Point Budget."""
    if widget_key not in st.session_state: return 
    val = st.session_state[widget_key]
    
    # 1. Check individual cap
    if val > max_val:
        val = int(max_val)
        st.toast(f"⚠️ **{item_name}** exceeds limit. Clamped to Max ({max_val}).")
    elif val < min_val:
        val = int(min_val)
        st.toast(f"⚠️ **{item_name}** below limit. Clamped to Min ({min_val}).")
        
    # 2. Check the Global Stat Budget (Arch Level + Upgrade 12)
    total_allowed = int(st.session_state.player.arch_level) + int(st.session_state.player.upgrade_levels.get(12, 0))
    
    other_sum = 0
    for s in st.session_state.player.base_stats.keys():
        if s != stat_key:
            # We must use session_state to grab the LIVE widget values in case multiple changed
            current = st.session_state.get(f"stat_{s}", st.session_state.player.base_stats.get(s, 0))
            other_sum += int(current)
            
    if val + other_sum > total_allowed:
        max_possible = max(0, total_allowed - other_sum)
        if val > max_possible:
            val = max_possible
            st.toast(f"⚠️ Not enough Stat Points! Clamped **{item_name}** to {val}.")
            
    st.session_state[widget_key] = val

def update_external_group(group_id, rows):
    """Callback to sync a UI widget value to all corresponding engine rows."""
    if group_id not in st.session_state: return
    val = st.session_state[group_id]
    for r in rows:
        st.session_state.player.set_external_level(r, int(val))

def update_card_level(widget_key, card_id):
    """Callback to sync a UI widget value directly to the Player's card inventory."""
    if widget_key not in st.session_state: return
    val = st.session_state[widget_key]
    st.session_state.player.set_card_level(card_id, int(val))

# --- IMAGE CENTERING & SCALING HELPERS ---
def render_centered_image(img_source, target_width):
    """
    Physically resizes the image using PIL Nearest Neighbor for razor-sharp 
    retro pixels, completely bypassing Streamlit CSS rendering bugs.
    """
    # 1. Load image into PIL
    if isinstance(img_source, str):
        img = Image.open(img_source).convert("RGBA")
    else:
        img = img_source
        
    # 2. Scale it physically in memory
    w_percent = (target_width / float(img.width))
    target_height = int((float(img.height) * float(w_percent)))
    img_resized = img.resize((target_width, target_height), Image.NEAREST)
    
    # 3. Convert to Base64
    buffered = BytesIO()
    img_resized.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
        
    # 4. Inject into centered HTML
    html = f"""
    <div style="display: flex; justify-content: center; margin-bottom: 10px;">
        <img src="data:image/png;base64,{encoded}">
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def composite_card(bg_path, core_path, y_offset):
    """Dynamically overlays ANY core asset onto a dynamic background."""
    try:
        bg = Image.open(bg_path).convert("RGBA")
        fg = Image.open(core_path).convert("RGBA")
        
        offset_x = (bg.width - fg.width) // 2
        # Apply the custom offset to shift the core up or down into the frame
        offset_y = ((bg.height - fg.height) // 2) + y_offset
        
        composite = bg.copy()
        composite.paste(fg, (offset_x, offset_y), mask=fg)
        return composite
    except Exception as e:
        return None

def find_external_image(upg_id):
    """Uses glob to find images with prefixes like '4_hestia.png'."""
    pattern = os.path.join(ROOT_DIR, "assets", "upgrades", "external", f"{upg_id}_*.png")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    exact = os.path.join(ROOT_DIR, "assets", "upgrades", "external", f"{upg_id}.png")
    return exact if os.path.exists(exact) else None

def get_scaled_image_uri(filepath, target_width):
    """Scales an image using NEAREST and returns a Base64 URI for Streamlit DataFrames."""
    if os.path.exists(filepath):
        img = Image.open(filepath).convert("RGBA")
        w_percent = (target_width / float(img.width))
        target_height = int((float(img.height) * float(w_percent)))
        img_resized = img.resize((target_width, target_height), Image.NEAREST)
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{encoded}"
    return None

# --- SESSION STATE INITIALIZATION ---
if 'player' not in st.session_state:
    st.session_state.player = Player()

p = st.session_state.player
st.set_page_config(page_title="AI Arch Optimizer", layout="wide", page_icon="⛏️")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    # --- 1. GLOBAL SETTINGS (Moved to top!) ---
    st.header("⚙️ Global Settings")
    
    # Initialize session state for global settings so they don't throw warnings
    if "set_asc2" not in st.session_state: st.session_state["set_asc2"] = p.asc2_unlocked
    if "set_arch" not in st.session_state: st.session_state["set_arch"] = int(p.arch_level)
    if "set_floor" not in st.session_state: st.session_state["set_floor"] = int(p.current_max_floor)
    if "set_hades" not in st.session_state: st.session_state["set_hades"] = int(p.hades_idol_level)
    
    # Render widgets with explicit keys
    p.asc2_unlocked = st.checkbox("Ascension 2 Unlocked", key="set_asc2")
    p.arch_level = st.number_input("Arch Level", min_value=1, step=1, key="set_arch")
    p.current_max_floor = st.number_input("Max Floor Reached", min_value=1, step=1, key="set_floor")
    
    if p.asc2_unlocked:
        p.hades_idol_level = st.number_input("Hades Idol Level", min_value=0, step=1, key="set_hades")
    else:
        p.hades_idol_level = 0

    st.divider()

    # --- 2. IMPORT DATA ---
    st.header("📂 Import Data")
    uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
    
    if uploaded_file is not None:
        # Prevent infinite reloading: Only process if it is a NEW file upload!
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.file_id:
            st.session_state.last_uploaded_file = uploaded_file.file_id
            
            temp_path = os.path.join(ROOT_DIR, "temp_upload.json")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            load_state_from_json(p, temp_path)
            os.remove(temp_path)
            
            # Flush ALL widget keys so they resync to the new JSON file!
            for k in list(st.session_state.keys()):
                if k.startswith(("upg_", "stat_", "ext_", "card_", "set_")):
                    del st.session_state[k]
                    
            # Force a clean restart from Line 1 to sync the reordered sidebar
            st.rerun() 
            
    st.divider()
    
    # --- 3. EXPORT DATA ---
    st.header("💾 Export Data")
    st.write("Download your current UI configuration.")
    
    temp_export = os.path.join(ROOT_DIR, "temp_export.json")
    save_state_to_json(p, temp_export, readable_keys=True, hide_locked=True)
    with open(temp_export, "r") as f:
        export_json_str = f.read()
    if os.path.exists(temp_export):
        os.remove(temp_export)
        
    st.download_button(
        label="📥 Download player_state.json",
        data=export_json_str,
        file_name="player_state.json",
        mime="application/json",
        use_container_width=True
    )


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

tab_stats, tab_upgrades, tab_cards, tab_calc_stats, tab_ore_stats, tab_optimizer = st.tabs([
    "📊 Base Stats", "⬆️ Upgrades", "🃏 Ore Cards", "🧮 Calculated Stats", "🪨 Ore Stats", "🚀 Run Optimizer"
])

# --- TAB 1: BASE STATS ---
with tab_stats:
    
    # --- GLOBAL STAT BUDGET TRACKER ---
    total_allowed = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
    current_allocated = sum(int(st.session_state.get(f"stat_{s}", p.base_stats.get(s, 0))) for s in p.base_stats.keys())
    remaining = total_allowed - current_allocated
    
    col_title, col_tracker = st.columns([2, 1])
    with col_title:
        st.subheader("Base Stat Allocation")
    with col_tracker:
        # A sleek Streamlit metric box to track points
        st.metric(
            label="Unallocated Points", 
            value=remaining, 
            delta=f"{current_allocated} / {total_allowed} Used",
            delta_color="off"
        )
        
    if remaining < 0:
        st.error(f"⚠️ You have over-allocated stats by {abs(remaining)} points! Please lower them before running the optimizer.")
    
    st.divider()

    def render_stat(label, stat_key):
        max_val = int(STAT_CAPS[stat_key])
        current_val = int(p.base_stats.get(stat_key, 0))
        safe_val = min(max(current_val, 0), max_val)
        widget_key = f"stat_{stat_key}"
        
        if widget_key not in st.session_state:
            st.session_state[widget_key] = safe_val
            
        with st.container(border=True):
            # Centered Title
            st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'><b>{label}</b><br><small>(Max: {max_val})</small></div>", unsafe_allow_html=True)
            
            # Centered Image
            img_path = os.path.join(ROOT_DIR, "assets", "stats", f"{stat_key.lower()}.png")
            if os.path.exists(img_path):
                render_centered_image(img_path, UI_STAT_IMG_WIDTH)
            else:
                st.markdown("<div style='text-align: center; color: gray;'><br><small>(Icon Missing)</small><br><br></div>", unsafe_allow_html=True)
            
            # Number Input
            st.number_input(
                f"{label} (Max: {max_val})",
                key=widget_key, step=1, on_change=enforce_stat_caps,
                args=(widget_key, stat_key, 0, max_val, label),
                label_visibility="collapsed"
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
            # Silently keep Corruption at 0 without a spoiler message
            p.base_stats['Corr'] = 0

# --- TAB 2: UPGRADES ---
with tab_upgrades:
    sub_internal, sub_external = st.tabs(["Internal Upgrades", "External Upgrades"])
    
    with sub_internal:
        asc2_locked_rows =[17, 19, 34, 46, 52, 55]
        
        # 1. Pre-filter active upgrades
        active_upgrades =[]
        for upg_id, upg_data in p.UPGRADE_DEF.items():
            if not p.asc2_unlocked and upg_id in asc2_locked_rows:
                continue
            active_upgrades.append((upg_id, upg_data))
            
        # UI TWEAK: Uses the ratio from the top of the file
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
                        # Internal upgrades can still use container width since they are in a heavily constrained column
                        st.image(img_path, width="stretch")
                    
                    st.number_input(
                        f"Level##int_{upg_id}", key=widget_key, step=1, 
                        on_change=enforce_caps, args=(widget_key, 0, max_lvl, name),
                        label_visibility="collapsed"
                    )
                    p.set_upgrade_level(upg_id, st.session_state[widget_key])

    with sub_external:
        # UI TWEAK: Create the full array of columns
        cols_all = st.columns(UI_EXT_COL_RATIO)
        
        # Slice off the first and last columns to act as empty buffers
        active_cols = cols_all[1:-1]
        num_active = len(active_cols)
        
        for idx, group in enumerate(cfg.EXTERNAL_UI_GROUPS):
            widget_key = f"ext_{group['id']}"
            ui_type = group['ui_type']
            rows = group['rows']
            
            current_val = int(p.external_levels.get(rows[0], 0))
            if widget_key not in st.session_state:
                st.session_state[widget_key] = current_val

            # Distribute items only into the active middle columns
            with active_cols[idx % num_active]:
                with st.container(border=True):
                    
                    # Centered Title for External Upgrades
                    st.markdown(f"<div style='text-align: center; margin-bottom: 10px;'><b>{group['name']}</b></div>", unsafe_allow_html=True)
                    
                    # --- ASSET LOADING WITH CENTERED BASE64 HTML ---
                    if ui_type == "skill":
                        for img_name in group.get("imgs",[]):
                            img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "external", img_name)
                            if os.path.exists(img_path):
                                # Differentiate between icon and text images
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
                            # Passed the explicit core path and offset here
                            core_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", "20_Misc_Arch_Ability_face.png")
                            comp_img = composite_card(bg_path, core_path, UI_EXT_CARD_CORE_Y_OFFSET)
                            
                            if comp_img:
                                render_centered_image(comp_img, UI_EXT_IMG_CARD)
                            else:
                                st.markdown("<div style='text-align: center; color: gray;'>(Card Assets Missing)</div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; color: gray;'>(Card Not Unlocked)</div>", unsafe_allow_html=True)

                    st.divider()

                    # --- WIDGET LOGIC ---
                    # Inject custom subtext for Geoduck
                    if group['id'] == 'geoduck':
                        st.markdown("<div style='text-align: center; color: gray; margin-top: -10px; margin-bottom: 10px;'><small>Enter Number of Mythic Chests Opened</small></div>", unsafe_allow_html=True)

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

# --- TAB 3: ORE CARDS ---
with tab_cards:
    st.subheader("Ore Card Collection")
    
    ore_types =['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']
    
    # Loop over the 4 Tiers (Rows)
    for tier_num in range(1, 5):
        # Create exactly 7 columns for the 7 Ore types
        cols_cards = st.columns(7)
        
        for col_idx, o_type in enumerate(ore_types):
            card_id = f"{o_type}{tier_num}"
            widget_key = f"card_{card_id}"
            
            current_lvl = int(p.cards.get(card_id, 0))
            if widget_key not in st.session_state:
                st.session_state[widget_key] = current_lvl
                
            with cols_cards[col_idx]:
                with st.container(border=True):
                    # Title
                    st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'><b>{card_id.capitalize()}</b></div>", unsafe_allow_html=True)
                    
                    user_tier = st.session_state[widget_key]
                    
                    # --- DYNAMIC CARD COMPOSITING ---
                    if user_tier > 0:
                        bg_path = os.path.join(ROOT_DIR, "assets", "cards", "backgrounds", f"{user_tier}.png")
                        core_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{card_id}.png")
                        
                        comp_img = composite_card(bg_path, core_path, UI_ORE_CARD_Y_OFFSET)
                        if comp_img:
                            render_centered_image(comp_img, UI_ORE_CARD_WIDTH)
                        else:
                            st.markdown("<div style='text-align: center; color: gray;'><small>(Assets Missing)</small></div><br>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='text-align: center; color: gray;'><br><small>(Not Unlocked)</small><br><br></div>", unsafe_allow_html=True)
                        
                    st.divider()
                    
                    # --- ASC2 LOCK LOGIC ---
                    is_locked = (tier_num == 4 and not p.asc2_unlocked)
                    
                    if is_locked:
                        st.markdown("<div style='text-align: center; color: #ff4b4b;'><small>Locked (Asc2)</small></div>", unsafe_allow_html=True)
                    else:
                        st.number_input(
                            f"Lvl##{card_id}", min_value=0, max_value=4,
                            key=widget_key, step=1,
                            on_change=update_card_level, args=(widget_key, card_id),
                            label_visibility="collapsed"
                        )
                        p.set_card_level(card_id, st.session_state[widget_key])

# --- TAB 4: CALCULATED STATS ---
with tab_calc_stats:
    st.subheader("Calculated Player Stats")
    st.write("This is the exact mathematical output derived from your Base Stats, Upgrades, and Cards being fed into the Engine.")
    
    col_calc_1, col_calc_2, col_calc_3 = st.columns(3)
    
    with col_calc_1:
        with st.container(border=True):
            st.markdown("#### ⚔️ Combat & Crits")
            st.write(f"**Max Stamina:** {p.max_sta:,.0f}")
            st.write(f"**Damage:** {p.damage:,.0f}")
            st.write(f"**Armor Pen:** {p.armor_pen:,.0f}")
            
            st.divider()
            
            # --- TRUE NESTED PROBABILITIES & COMPOUND MULTIPLIERS ---
            true_reg = 1.0 - p.crit_chance
            true_crit = p.crit_chance * (1.0 - p.super_crit_chance)
            true_scrit = p.crit_chance * p.super_crit_chance * (1.0 - p.ultra_crit_chance)
            true_ucrit = p.crit_chance * p.super_crit_chance * p.ultra_crit_chance
            
            comp_crit = p.crit_dmg_mult
            comp_scrit = p.crit_dmg_mult * p.super_crit_dmg_mult
            comp_ucrit = p.crit_dmg_mult * p.super_crit_dmg_mult * p.ultra_crit_dmg_mult
            
            st.markdown("#### 🎯 True Hit Breakdown")
            st.write(f"*- Regular:* {true_reg*100:,.2f}%")
            st.write(f"*- Crit:* {true_crit*100:,.2f}% *(Mult: {comp_crit:,.2f}x)*")
            st.write(f"*- Super Crit:* {true_scrit*100:,.2f}% *(Mult: {comp_scrit:,.2f}x)*")
            st.write(f"*- Ultra Crit:* {true_ucrit*100:,.2f}% *(Mult: {comp_ucrit:,.2f}x)*")
            
            st.divider()
            st.markdown("<small>*Raw Stats (Before Nesting)*</small>", unsafe_allow_html=True)
            st.write(f"<small>Base Crit: {p.crit_chance*100:.2f}% | sCrit: {p.super_crit_chance*100:.2f}% | uCrit: {p.ultra_crit_chance*100:.2f}%</small>", unsafe_allow_html=True)

    with col_calc_2:
        with st.container(border=True):
            st.markdown("#### 💰 Economy & Modifiers")
            st.write(f"**EXP Gain Multiplier:** {p.exp_gain_mult:,.2f}x")
            st.write(f"**Frag/Loot Multiplier:** {p.frag_loot_gain_mult:,.2f}x")
            st.divider()
            st.write(f"**EXP Mod Chance:** {p.exp_mod_chance*100:,.2f}% *(Mod Multi: {p.exp_mod_gain:,.2f}x)*")
            st.write(f"**Loot Mod Chance:** {p.loot_mod_chance*100:,.2f}% *(Mod Multi: {p.loot_mod_gain:,.2f}x)*")
            st.write(f"**Stamina Mod Chance:** {p.stamina_mod_chance*100:,.2f}% *(Mod Gain: +{p.stamina_mod_gain:,.0f} Stamina)*")
            st.write(f"**Speed Mod Chance:** {p.speed_mod_chance*100:,.2f}% *(Mod Gain: +{p.speed_mod_gain:,.0f} 2x spd atks)*")
            st.divider()
            st.write(f"**Crosshair Auto-Tap Chance:** {p.crosshair_auto_tap*100:,.2f}%")
            st.write(f"**Gold Crosshair Chance:** {p.gold_crosshair_chance*100:,.2f}% *(Mult: {p.gold_crosshair_mult:,.2f}x)*")

    with col_calc_3:
        with st.container(border=True):
            st.markdown("#### ⚡ Abilities")
            st.write(f"**Instacharge Chance:** {p.ability_insta_charge*100:,.2f}%")
            st.divider()
            st.write(f"**Enrage:** {p.enrage_charges:,.0f} charges *(CD: {p.enrage_cooldown:,.1f}s)*")
            st.write(f"*- Dmg Bonus:* +{p.enrage_bonus_dmg*100:,.0f}%")
            st.write(f"*- Enraged Dmg:* {p.enraged_damage:,.0f}")
            st.write(f"*- Crit Bonus:* +{p.enrage_bonus_crit_dmg*100:,.0f}%")
            st.write(f"*- Enraged Crit Dmg:* {p.enraged_crit_dmg_mult:,.2f}x")
            st.divider()
            st.write(f"**Flurry:** {p.flurry_duration:,.0f}s *(CD: {p.flurry_cooldown:,.1f}s)*")
            st.write(f"*- Stamina Gain:* {p.flurry_sta_on_cast:,.0f}")
            st.write(f"*- +100% Atk Speed")
            st.divider()
            st.write(f"**Quake:** {p.quake_attacks:,.0f} atks *(CD: {p.quake_cooldown:,.1f}s)*")
            st.write(f"*- Splash Dmg:* {p.quake_dmg_to_all*100:,.0f}%")
        
        # Conditionally render the endgame variables!
        if p.asc2_unlocked:
            with st.container(border=True):
                st.markdown("#### 🌌 Ascension 2")
                st.write(f"**Gleaming Chance:** {p.gleaming_floor_chance*100:,.2f}%")
                st.write(f"**Gleaming Multiplier:** {p.gleaming_floor_multi:,.2f}x")
                st.write(f"**Infernal Multiplier:** {p.infernal_multiplier:,.4f}x")

# --- TAB 5: ORE STATS ---
with tab_ore_stats:
    st.subheader("Ore Compendium")
    
    col_ore_toggle, col_ore_floor = st.columns([1, 1])
    with col_ore_toggle:
        show_modified = st.toggle("Show Modified Stats (Applies player multipliers, cards, and floor scaling)")
    
    target_floor = 1
    if show_modified:
        with col_ore_floor:
            target_floor = st.number_input("Calculate scaling for Floor Level:", min_value=1, value=int(p.current_max_floor), step=1)
            
    st.divider()

    FRAG_NAMES = {0: "Dirt", 1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Mythic", 6: "Divinity"}
    table_data =[]
    
    for ore_id, base in cfg.ORE_BASE_STATS.items():
        # Hide Tier 4 ores if Asc2 is not unlocked
        if not p.asc2_unlocked and ore_id.endswith('4'):
            continue
            
        img_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{ore_id}.png")
        img_uri = get_scaled_image_uri(img_path, UI_ORE_TABLE_IMG_WIDTH)
        frag_name = FRAG_NAMES.get(base.get('ft', 0), "Unknown")
        
        if show_modified:
            # Feed it into the Layer 2 engine to get exact scaled math
            ore_obj = Ore(ore_id, target_floor, p)
            
            # Apply Player Armor Penetration to the scaled armor!
            eff_armor = max(0, ore_obj.armor - p.armor_pen)
            
            table_data.append({
                "Icon": img_uri,
                "Ore": ore_id.capitalize(),
                "HP": f"{ore_obj.hp:,}",
                "Eff. Armor": f"{eff_armor:,.0f} (Base: {ore_obj.armor:,})",
                "XP Yield": f"{ore_obj.xp:,.2f}",
                "Frag Yield": f"{ore_obj.frag_amt:,.3f}",
                "Frag Type": frag_name
            })
        else:
            # Just show the raw dictionary values
            table_data.append({
                "Icon": img_uri,
                "Ore": ore_id.capitalize(),
                "Base HP": f"{base['hp']:,}",
                "Base Armor": f"{base['a']:,}",
                "Base XP": f"{base['xp']:,.2f}",
                "Base Frags": f"{base['fa']:,.3f}",
                "Frag Type": frag_name
            })
            
    # Render the interactive Streamlit dataframe
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            column_config={
                "Icon": st.column_config.ImageColumn("Icon", help="Ore Icon"),
            },
            hide_index=True,
            width="stretch",
            height=600 # Makes the table nice and tall so you don't have to scroll constantly
        )

# --- TAB 6: RUN OPTIMIZER ---
with tab_optimizer:
    st.header("🚀 Monte Carlo Stat Optimizer")
    st.write("Leverage Successive Halving to find the absolute mathematically perfect stat distribution. Ensure your total allocated points do not exceed your budget before running.")

    # --- GOAL SELECTION ---
    col_goal, col_target = st.columns(2)
    with col_goal:
        opt_goal = st.selectbox(
            "Optimization Target",["Max Floor Push", "Max EXP Yield", "Fragment Farming", "Card/Ore Farming"]
        )
    
    with col_target:
        target_metric = "highest_floor" # Default fallback
        if opt_goal == "Fragment Farming":
            frag_tier = st.selectbox(
                "Fragment Tier",[0, 1, 2, 3, 4, 5, 6], 
                format_func=lambda x: {0:"Dirt", 1:"Common", 2:"Rare", 3:"Epic", 4:"Legendary", 5:"Mythic", 6:"Divine"}.get(x)
            )
            target_metric = f"frag_{frag_tier}_per_min"
        elif opt_goal == "Card/Ore Farming":
            ore_target = st.text_input("Target Ore ID (e.g., com1, myth3)", value="myth3").lower()
            target_metric = f"ore_{ore_target}_per_min"
        elif opt_goal == "Max EXP Yield":
            target_metric = "xp_per_min"
        else:
            target_metric = "highest_floor"

    st.divider()

    # --- HARDWARE BENCHMARKING & ETA ---
    if "sims_per_sec" not in st.session_state:
        st.session_state.sims_per_sec = 0
        st.session_state.eta_profiles = {}

    with st.expander("🧠 How does the AI Optimizer work? (Click to read)"):
        st.markdown("""
        **1. The 3-Phase "Zoom-In" Grid Search:**
        Testing every possible stat combination point-by-point would require millions of simulations and take days. Instead, we "zoom in":
        * **Phase 1 (Coarse):** We cast a wide net across your entire stat budget in large leaps (e.g., leaps of 10 points) to find the general neighborhood of the optimal build.
        * **Phase 2 (Fine):** We draw a tight box around the Phase 1 winner and test smaller leaps (e.g., leaps of 3 points).
        * **Phase 3 (Exact):** We draw a final box around the Phase 2 winner and test *every single point* (leaps of 1) to find the mathematical peak.
        
        **2. Successive Halving (Early Culling):**
        During each phase, we don't test bad builds thoroughly. We test all builds briefly (15 runs), immediately delete the bottom 80% of performers, test the survivors a bit more (35 runs), and reserve the heaviest testing purely for the top contenders.
        """)

    col_bench, col_prof = st.columns([1, 1.5])
    
    with col_bench:
        st.write("#### 1. Hardware Benchmark")
        st.write("*(Optional: Runs automatically on start if skipped)*")
        if st.button("⏱️ Benchmark CPU & Calculate ETAs", use_container_width=True):
            with st.spinner("Running 200 micro-simulations to test CPU speed..."):
                # Construct baseline payload
                STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                
                payload = {'stats': {s: int(p.base_stats.get(s, 0)) for s in STATS_TO_OPTIMIZE}, 'fixed_stats': {}}
                CPU_CORES = max(1, mp.cpu_count() - 1)
                
                with mp.Pool(CPU_CORES) as pool:
                    spd = benchmark_hardware(payload, pool)
                    st.session_state.sims_per_sec = spd
                    
                    # Calculate profiles
                    budget = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
                    cap_increase = int(p.u('H45'))
                    caps = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in STATS_TO_OPTIMIZE}
                    
                    st.session_state.eta_profiles = get_eta_profiles(STATS_TO_OPTIMIZE, budget, caps, spd)
        
        if st.session_state.sims_per_sec > 0:
            st.success(f"⚡ **Hardware Speed:** {st.session_state.sims_per_sec:,.0f} simulations / second")
        else:
            st.info("Awaiting Benchmark...")

    with col_prof:
        st.write("#### 2. Search Depth (Initial Step Size)")
        
        # Transparent labels that show exactly what the knob is doing
        depth_labels = {
            "Fast": "Fast (Step 15) - Best for quick checks",
            "Standard": "Standard (Step 10) - Recommended balance",
            "Deep": "Deep (Step 5) - Exhaustive, takes much longer"
        }
        
        depth_choice = st.radio(
            "Select Search Depth", 
            options=list(depth_labels.keys()), 
            format_func=lambda x: depth_labels[x],
            horizontal=False, 
            label_visibility="collapsed"
        )
        
        # Derive the exact steps that will be used based on the choice
        step_1 = {"Fast": 15, "Standard": 10, "Deep": 5}[depth_choice]
        step_2 = max(2, step_1 // 3)
        step_3 = 1
        
        # Build the dynamic preview box
        preview_html = f"""
        <div style='font-size: 0.9em; padding: 10px; border-left: 3px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1); margin-top: 10px;'>
            <b>Engine Execution Plan:</b><br>
            🔍 <b>Phase 1:</b> Scanning grid in leaps of <b>{step_1}</b>...<br>
            🔎 <b>Phase 2:</b> Zooming in with leaps of <b>{step_2}</b>...<br>
            🎯 <b>Phase 3:</b> Pinpointing exact peak with leaps of <b>{step_3}</b>.
        """
        
        if st.session_state.eta_profiles:
            prof_key = next(k for k in st.session_state.eta_profiles.keys() if k.startswith(depth_choice))
            prof_data = st.session_state.eta_profiles[prof_key]
            
            # Append the ETA inside the dynamic box
            preview_html += f"<br><br>⏱️ <b>Estimated Time:</b> {prof_data['time_label']} <i>(~{prof_data['builds']:,.0f} unique builds tested)</i>"
        
        preview_html += "</div>"
        st.markdown(preview_html, unsafe_allow_html=True)

    st.divider()

    # --- MONTE CARLO EXECUTION LOOP ---
    if st.button("🚀 Run Optimizer", use_container_width=True, type="primary"):
        st.write("---")
        
        # AUTO-BENCHMARK FAILSAFE
        if st.session_state.sims_per_sec == 0:
            with st.spinner("⏱️ First-time setup: Benchmarking your CPU..."):
                STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                payload = {'stats': {s: int(p.base_stats.get(s, 0)) for s in STATS_TO_OPTIMIZE}, 'fixed_stats': {}}
                CPU_CORES = max(1, mp.cpu_count() - 1)
                with mp.Pool(CPU_CORES) as pool:
                    spd = benchmark_hardware(payload, pool)
                    st.session_state.sims_per_sec = spd
                    budget = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
                    cap_increase = int(p.u('H45'))
                    caps = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in STATS_TO_OPTIMIZE}
                    st.session_state.eta_profiles = get_eta_profiles(STATS_TO_OPTIMIZE, budget, caps, spd)

        # Print ETA to screen before starting the heavy load
        prof_key = next((k for k in st.session_state.eta_profiles.keys() if k.startswith(depth_choice)), None)
        if prof_key:
            prof_data = st.session_state.eta_profiles[prof_key]
            step_size = prof_data['step']
            st.info(f"⏱️ **Running {depth_choice} Search:** Estimated to take {prof_data['time_label']} (~{prof_data['builds']:,.0f} builds at {st.session_state.sims_per_sec:,.0f} sims/sec)")

        with st.spinner(f"Engine Running..."):
            start_time = time.time()
            
            # 1. Prepare Engine Data
            STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
            if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
            
            DYNAMIC_BUDGET = int(sum(st.session_state.get(f"stat_{s}", p.base_stats.get(s, 0)) for s in STATS_TO_OPTIMIZE))
            FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
            cap_increase = int(p.u('H45'))
            EFFECTIVE_CAPS = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in STATS_TO_OPTIMIZE}
            
            CPU_CORES = max(1, mp.cpu_count() - 1)
            ITERATIONS_PER_DIST = 100
            
            best_p3, final_summary = None, None
            
            # --- LIVE UI PROGRESS BAR BRIDGE ---
            ui_prog_bar = st.progress(0, text="Booting up engine cores...")
            def st_progress_callback(phase_name, r_idx, r_total, task_idx, task_total):
                pct = min(100, max(0, int((task_idx / task_total) * 100)))
                ui_prog_bar.progress(pct, text=f"⚙️ {phase_name} | Round {r_idx}/{r_total} | {pct}% ({task_idx}/{task_total} sims completed)")
            
            # 2. Trigger Parallel Worker Pool
            with mp.Pool(CPU_CORES) as pool:
                # Phase 1: Coarse Grid
                bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
                best_p1, _ = run_optimization_phase(
                    "Phase 1 (Coarse)", target_metric, STATS_TO_OPTIMIZE, 
                    DYNAMIC_BUDGET, step_size, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p1,
                    progress_callback=st_progress_callback
                )
                
                if best_p1:
                    # Phase 2: Fine Grid
                    bounds_p2 = {s: (max(0, best_p1[s] - step_size), min(EFFECTIVE_CAPS[s], best_p1[s] + step_size)) for s in STATS_TO_OPTIMIZE}
                    step_2 = max(2, step_size // 3)
                    best_p2, _ = run_optimization_phase(
                        "Phase 2 (Fine)", target_metric, STATS_TO_OPTIMIZE, 
                        DYNAMIC_BUDGET, step_2, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p2,
                        progress_callback=st_progress_callback
                    )
                    
                    # Phase 3: Exact Micro-Grid
                    bounds_p3 = bounds_p2
                    if best_p2:
                        # Cap the Phase 3 radius to a max of 2 points.
                        # This prevents "Fast Mode" from exploding into 160k simulations!
                        p3_radius = min(2, step_2) 
                        bounds_p3 = {s: (max(0, best_p2[s] - p3_radius), min(EFFECTIVE_CAPS[s], best_p2[s] + p3_radius)) for s in STATS_TO_OPTIMIZE}
                    
                    best_p3, final_summary = run_optimization_phase(
                        "Phase 3 (Exact)", target_metric, STATS_TO_OPTIMIZE, 
                        DYNAMIC_BUDGET, 1, ITERATIONS_PER_DIST, pool, FIXED_STATS, bounds_p3,
                        progress_callback=st_progress_callback
                    )
            
            ui_prog_bar.empty() # Remove the progress bar cleanly when done!
            elapsed = time.time() - start_time
            
            # --- 3. UI RESULTS TELEMETRY ---
            if best_p3 and final_summary:
                st.success(f"✅ Successive Halving Complete in {elapsed:.1f} seconds!")
                
                res_col1, res_col2 = st.columns([1.5, 1])
                
                with res_col1:
                    with st.container(border=True):
                        st.markdown("#### 🏆 Optimal Stat Build")
                        # Beautiful Interactive Plotly Bar Chart
                        df_stats = pd.DataFrame({
                            "Stat": list(best_p3.keys()), 
                            "Points": list(best_p3.values())
                        })
                        fig = px.bar(
                            df_stats, x="Stat", y="Points", text="Points", 
                            color="Stat", color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_traces(textposition='outside')
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Auto-Apply UX Injection
                        if st.button("✨ Apply Build to UI", use_container_width=True):
                            for k, v in best_p3.items():
                                st.session_state[f"stat_{k}"] = int(v)
                                p.base_stats[k] = int(v)
                            st.rerun()

                with res_col2:
                    with st.container(border=True):
                        st.markdown("#### 📈 Projected Return")
                        val = final_summary[target_metric]
                        
                        if target_metric == "highest_floor":
                            st.metric("Max Floor Reached", f"{val:,.1f}")
                        else:
                            rate_sec = val / 60.0
                            rate_1k = rate_sec * 1000.0
                            
                            # Clean up UI string for Target Ore mapping
                            if "frag" in target_metric: metric_str = "Fragments"
                            elif "ore" in target_metric: metric_str = "Kills"
                            else: metric_str = "EXP"

                            st.metric(f"Real-Time ({metric_str})", f"{val:,.2f} / min")
                            st.divider()
                            st.metric("Banked Time", f"{rate_sec:,.2f} / Arch Sec")
                            st.metric("Banked Time (1k)", f"{rate_1k:,.1f} / 1k Arch Sec")
            else:
                st.error("Optimization failed. Could not generate valid builds for this stat distribution/budget.")