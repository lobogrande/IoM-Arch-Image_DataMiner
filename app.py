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
from io import BytesIO
from PIL import Image

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
# ==============================================================================
# ==============================================================================

# --- PATH RESOLUTION ---
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

from core.player import Player
from tools.verify_player import load_state_from_json, save_state_to_json
import project_config as cfg

# --- AUTO-CLAMPING CALLBACKS ---
def enforce_caps(key, min_val, max_val, item_name):
    """Standard clamping for independent limits (e.g. Upgrade Levels)."""
    val = st.session_state[key]
    if val > max_val:
        st.session_state[key] = int(max_val)
        st.toast(f"⚠️ **{item_name}** exceeds limit. Clamped to Max ({max_val}).")
    elif val < min_val:
        st.session_state[key] = int(min_val)
        st.toast(f"⚠️ **{item_name}** below limit. Clamped to Min ({min_val}).")

def enforce_stat_caps(widget_key, stat_key, min_val, max_val, item_name):
    """Specialized clamping for Base Stats that also checks the Global Point Budget."""
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
    val = st.session_state[group_id]
    for r in rows:
        st.session_state.player.set_external_level(r, int(val))

def update_card_level(widget_key, card_id):
    """Callback to sync a UI widget value directly to the Player's card inventory."""
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

# --- SESSION STATE INITIALIZATION ---
if 'player' not in st.session_state:
    st.session_state.player = Player()

p = st.session_state.player
st.set_page_config(page_title="AI Arch Optimizer", layout="wide", page_icon="⛏️")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("📂 Import Data")
    uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
    
    if uploaded_file is not None:
        temp_path = os.path.join(ROOT_DIR, "temp_upload.json")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_state_from_json(p, temp_path)
        os.remove(temp_path)
        
        # Flush existing widget keys so they resync to the new JSON file!
        for k in list(st.session_state.keys()):
            if k.startswith("upg_") or k.startswith("stat_") or k.startswith("ext_") or k.startswith("card_"):
                del st.session_state[k]
        st.success("Save file loaded!")
    
    st.divider()
    
    # --- EXPORT FUNCTIONALITY ---
    st.header("💾 Export Data")
    st.write("Download your current UI configuration.")
    
    # We write the current memory state to a temporary file, read it as a string, and pass it to Streamlit
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
    
    st.divider()
    
    st.header("⚙️ Global Settings")
    p.asc2_unlocked = st.checkbox("Ascension 2 Unlocked", value=p.asc2_unlocked)
    p.arch_level = st.number_input("Arch Level", min_value=1, value=int(p.arch_level), step=1)
    p.current_max_floor = st.number_input("Max Floor Reached", min_value=1, value=int(p.current_max_floor), step=1)
    
    if p.asc2_unlocked:
        p.hades_idol_level = st.number_input("Hades Idol Level", min_value=0, value=int(p.hades_idol_level), step=1)
    else:
        p.hades_idol_level = 0

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
    "📊 Base Stats", "⬆️ Upgrades", "🃏 Ore Cards", "🚀 Run Optimizer"
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
                        st.image(img_path, use_container_width=True)
                    
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


with tab_optimizer:
    st.subheader("Target Optimization")
    st.write("Optimizer hooks coming soon...")