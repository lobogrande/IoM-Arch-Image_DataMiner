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
import hashlib
import multiprocessing as mp
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from collections import Counter
from PIL import Image
import pandas as pd

# ==============================================================================
# 🎨 UI TWEAK PANEL 🎨
# Adjust these numbers, hit Save, and watch your browser instantly update!
# ==============================================================================

# --- BASE STATS ---
# Width of the stat icons
UI_STAT_IMG_WIDTH = 220

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
# X: Negative moves left, Positive moves right. Y: Negative moves up, Positive moves down.
UI_EXT_CARD_CBLOCK_X_OFFSET = 0
UI_EXT_CARD_CBLOCK_Y_OFFSET = -4 

# --- NEW: CORE SCALING ---
# Shrinks the inner Block/Core image BEFORE it gets pasted onto the background.
# 1.0 = 100% (Original Size), 0.8 = 80%, 0.75 = 75%, etc.
UI_CARD_CBLOCK_SCALE = 0.7

# --- BLOCK CARDS ---
# Width of the generated cards in the 4x7 grid
UI_BLOCK_CARD_WIDTH = 100
# Offsets specifically for Block Card cores
UI_BLOCK_CARD_X_OFFSET = 1
UI_BLOCK_CARD_Y_OFFSET = -4

# Width of the block icons inside the Block Stats DataFrame table
UI_BLOCK_TABLE_IMG_WIDTH = 40
# ==============================================================================
# ==============================================================================

# --- PATH RESOLUTION ---
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

from core.player import Player
from core.block import Block
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

def composite_card(bg_path, cblock_path, x_offset, y_offset):
    """Dynamically overlays ANY core asset onto a dynamic background."""
    try:
        bg = Image.open(bg_path).convert("RGBA")
        fg = Image.open(cblock_path).convert("RGBA")
        
        # --- NEW: Scale the inner core down before pasting ---
        if UI_CARD_CBLOCK_SCALE != 1.0:
            new_w = max(1, int(fg.width * UI_CARD_CBLOCK_SCALE))
            new_h = max(1, int(fg.height * UI_CARD_CBLOCK_SCALE))
            fg = fg.resize((new_w, new_h), Image.NEAREST)
        
        # Apply the custom offsets to nudge the core into the perfect spot
        offset_x = ((bg.width - fg.width) // 2) + x_offset
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

# ==============================================================================
# 🛡️ MULTIPROCESSING GUARD (NEW)
# ==============================================================================
if __name__ == "__main__":

    # --- SESSION STATE INITIALIZATION ---
    if 'player' not in st.session_state:
        st.session_state.player = Player()

    p = st.session_state.player

    # Change "expanded" to "collapsed" if you want the sidebar to be completely hidden by default!
    st.set_page_config(
        page_title="AI Arch Optimizer", 
        layout="wide", 
        page_icon="⛏️", 
        initial_sidebar_state="expanded" 
    )

    # --- GLOBAL CSS OVERRIDES (REACT-STYLE UI MAPPING) ---
    st.markdown("""
        <style>
        /* 1. Force the collapsed 'open' arrow to be permanently visible and mid-screen */
        [data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            opacity: 1 !important;
            visibility: visible !important;
            background-color: #2b2b2b !important;
            border: 1px solid #ffa229 !important;
            border-left: none !important;
            border-radius: 0 8px 8px 0 !important;
            box-shadow: 2px 0px 5px rgba(0,0,0,0.5) !important;
            top: 50vh !important; 
            position: fixed !important;
            transform: translateY(-50%) !important;
            z-index: 100000 !important;
            transition: background-color 0.2s ease !important;
        }[data-testid="collapsedControl"]:hover, [data-testid="stSidebarCollapsedControl"]:hover {
            background-color: #ffa229 !important;
        }
        [data-testid="collapsedControl"] svg, [data-testid="stSidebarCollapsedControl"] svg {
            fill: #ffa229 !important;
            width: 20px !important;
            height: 20px !important;
        }
        [data-testid="collapsedControl"]:hover svg,[data-testid="stSidebarCollapsedControl"]:hover svg {
            fill: #2b2b2b !important;
        }
        
        /* 2. Highlight the 'close' button inside the expanded sidebar */[data-testid="stSidebarHeader"] button, [data-testid="baseButton-header"] {
            opacity: 1 !important;
            visibility: visible !important;
            background-color: rgba(255, 162, 41, 0.1) !important;
            border: 1px solid #ffa229 !important;
            border-radius: 6px !important;
        }[data-testid="stSidebarHeader"] button:hover, [data-testid="baseButton-header"]:hover {
            background-color: #ffa229 !important;
        }
        
        /* 3. Add a visual hint to the vertical resizer divider */[data-testid="stSidebarResizer"] {
            background-color: rgba(255, 162, 41, 0.2) !important;
            width: 4px !important;
        }
        [data-testid="stSidebarResizer"]:hover {
            background-color: rgba(255, 162, 41, 0.8) !important;
        }
        
        /* =========================================================
        4. GLOBAL UX COLORS & VERTICAL SPACING COMPRESSION
        ========================================================= */
        
        /* Pull the entire app higher up on the screen */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Squash the massive blank space under the 100/100 Metric */[data-testid="stMetric"] {
            margin-bottom: -15px !important;
        }
        
        /* Squash the massive margins around the st.divider() lines */
        hr {
            margin-top: 5px !important;
            margin-bottom: 15px !important;
        }
        
        /* Tighten the internal padding of the bordered stat containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.75rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 🔐 BETA ACCESS GATE (WITH URL MEMORY)
    # ==========================================
    # Securely grab the password from .streamlit/secrets.toml (locally) or the Streamlit Dashboard
    try:
        CORRECT_KEY = st.secrets["BETA_KEY"]
    except FileNotFoundError:
        st.error("Missing secrets configuration! App locked.")
        st.stop()

    # Generate a one-way cryptographic hash of the password
    master_hash = hashlib.sha256(CORRECT_KEY.encode()).hexdigest()

    # 1. Check if their URL contains the matching hash (survives server reboots!)
    if st.query_params.get("beta") == master_hash:
        st.session_state.beta_authorized = True
    elif "beta_authorized" not in st.session_state:
        st.session_state.beta_authorized = False

    if not st.session_state.beta_authorized:
        st.title("⛏️ IoM Arch Optimizer (Closed Beta)")
        st.warning("This application performs heavy Monte Carlo simulations. To prevent server overload during the testing phase, access is currently restricted.")
        
        # Removed type="password" to fix Firefox copy/paste blocking
        user_key = st.text_input("Enter Beta Key:")
        
        if st.button("Unlock Optimizer"):
            if user_key == CORRECT_KEY:
                st.session_state.beta_authorized = True
                # 2. Inject the secure hash into their browser URL
                st.query_params["beta"] = master_hash
                st.rerun()
            else:
                st.error("❌ Invalid Beta Key.")
                
        # Stop the script entirely so the rest of the app doesn't render or execute
        st.stop()

    # ==========================================
    # SIDEBAR
    # ==========================================
    with st.sidebar:
        
        # --- 1. GLOBAL SETTINGS ---
        with st.expander("⚙️ Global Settings", expanded=True):
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

        # --- 2. IMPORT DATA ---
        with st.expander("📂 Import Data", expanded=False):
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
                        if k.startswith(("upg_", "stat_", "ext_", "card_", "set_", "sandbox_")):
                            del st.session_state[k]
                            
                    # Force a clean restart from Line 1 to sync the reordered sidebar
                    st.rerun() 
            
        # --- 3. EXPORT DATA ---
        with st.expander("💾 Export Data", expanded=False):
            st.write("Download your current UI configuration.")
            
            # --- SYNC STATE BEFORE EXPORT ---
            for k, v in st.session_state.items():
                if k.startswith("stat_") and "sandbox" not in k:
                    stat_name = k.split("_")[1]
                    if stat_name in p.base_stats:
                        p.base_stats[stat_name] = int(v)
                elif k.startswith("upg_"):
                    try:
                        upg_id = int(k.split("_")[1])
                        p.set_upgrade_level(upg_id, int(v))
                    except ValueError:
                        pass

            temp_export = os.path.join(ROOT_DIR, "temp_export.json")
            save_state_to_json(p, temp_export, readable_keys=True, hide_locked=True)
            with open(temp_export, "r") as f:
                export_json_str = f.read()
            if os.path.exists(temp_export):
                os.remove(temp_export)
                
            st.download_button(
                label="📥 Download JSON",
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

    tab_stats, tab_upgrades, tab_cards, tab_calc_stats, tab_block_stats, tab_sandbox, tab_optimizer = st.tabs([
        "📊 Base Stats", "⬆️ Upgrades", "🎴 Block Cards", "📋 Calculated Player Stats", "🪨 Block Stats", "🧪 Block Hit Sandbox", "🚀 Run Optimizer"
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
            render_stat("Divine", 'Div')
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
            asc2_locked_rows =[19, 27, 34, 46, 52, 55]
            
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
                                cblock_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", "20_Misc_Arch_Ability_face.png")
                                comp_img = composite_card(bg_path, cblock_path, UI_EXT_CARD_CBLOCK_X_OFFSET, UI_EXT_CARD_CBLOCK_Y_OFFSET)
                                
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
                            
                            # --- INFERNAL BONUS DYNAMIC UI ---
                            # If this is the Arch Ability Card (Row 20) and it is set to Tier 4
                            if 20 in rows and st.session_state[widget_key] == 4:
                                st.markdown("<div style='text-align: center; margin-top: 5px; color: #ff4b4b;'><small><b>Infernal Cooldown Bonus %</b></small></div>", unsafe_allow_html=True)
                                
                                inf_key = f"ext_inf_{group['id']}"
                                if inf_key not in st.session_state:
                                    # Convert decimal to percentage for UI (e.g. -0.125 -> -12.5)
                                    st.session_state[inf_key] = float(p.arch_ability_infernal_bonus * 100.0)
                                    
                                def update_inf(k=inf_key):
                                    p.arch_ability_infernal_bonus = st.session_state[k] / 100.0
                                    p.set_external_level(20, 4) # Trigger W20 math refresh!
                                    
                                st.number_input(
                                    "Inf Bonus", max_value=0.0, step=0.1, format="%.2f",
                                    key=inf_key, on_change=update_inf, label_visibility="collapsed"
                                )

    # --- TAB 3: BLOCK CARDS ---
    with tab_cards:
        # --- CUSTOM DIV1 HEADER ICON ---
        bg_path = os.path.join(ROOT_DIR, "assets", "cards", "backgrounds", "1.png")
        cblock_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", "div1.png")
        
        comp_img = composite_card(bg_path, cblock_path, UI_BLOCK_CARD_X_OFFSET, UI_BLOCK_CARD_Y_OFFSET)
        if comp_img:
            # Scale down to a crisp header icon size (e.g., 40px wide)
            target_width = 40
            w_pct = (target_width / float(comp_img.width))
            target_height = int((float(comp_img.height) * float(w_pct)))
            img_resized = comp_img.resize((target_width, target_height), Image.NEAREST)
            
            buffered = BytesIO()
            img_resized.save(buffered, format="PNG")
            encoded = base64.b64encode(buffered.getvalue()).decode()
            
            # Inject seamlessly alongside the title text
            icon_html = f'<img src="data:image/png;base64,{encoded}" style="vertical-align: middle; margin-right: 12px; margin-bottom: 6px; border-radius: 4px;">'
            st.markdown(f"<h3>{icon_html}Block Card Collection</h3>", unsafe_allow_html=True)
        else:
            # Fallback if assets are missing
            st.subheader("🎴 Block Card Collection")
            
        block_types =['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']
        
        # Loop over the 4 Tiers (Rows)
        for tier_num in range(1, 5):
            
            # --- ASC2 LOCK LOGIC: Completely hide the 4th row to prevent spoilers ---
            if tier_num == 4 and not p.asc2_unlocked:
                continue
                
            # Create exactly 7 columns for the 7 Block types
            cols_cards = st.columns(7)
            
            for col_idx, o_type in enumerate(block_types):
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
                            cblock_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{card_id}.png")
                            
                            # Passing the new X Offset!
                            comp_img = composite_card(bg_path, cblock_path, UI_BLOCK_CARD_X_OFFSET, UI_BLOCK_CARD_Y_OFFSET)
                            if comp_img:
                                render_centered_image(comp_img, UI_BLOCK_CARD_WIDTH)
                            else:
                                st.markdown("<div style='text-align: center; color: gray;'><small>(Assets Missing)</small></div><br>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='text-align: center; color: gray;'><br><small>(Not Unlocked)</small><br><br></div>", unsafe_allow_html=True)
                            
                        st.divider()
                        
                        # Render native input without the spoiler red text logic
                        st.number_input(
                            f"Lvl##{card_id}", min_value=0, max_value=4,
                            key=widget_key, step=1,
                            on_change=update_card_level, args=(widget_key, card_id),
                            label_visibility="collapsed"
                        )
                        p.set_card_level(card_id, st.session_state[widget_key])

    # --- TAB 4: CALCULATED STATS ---
    with tab_calc_stats:
        st.subheader("📋 Calculated Player Stats")
        st.write("This is the exact mathematical output derived from your Base Stats, Upgrades, and Cards being fed into the Engine.")
        
        col_calc_1, col_calc_2, col_calc_3 = st.columns(3)
        
        with col_calc_1:
            with st.container(border=True):
                st.markdown("#### ⚔️ Combat & Crits")
                st.write(f"**Max Stamina:** {p.max_sta:,.0f}")
                st.write(f"**Damage:** {p.damage:,.0f}")
                st.write(f"**Armor Pen:** {p.armor_pen:,.0f}")
                
                st.divider()
                
                st.markdown("#### 📊 Raw In-Game Stats")
                st.write("<small><i>These are the exact numbers shown on your in-game UI screen. Verify these match!</i></small>", unsafe_allow_html=True)
                st.write(f"**Base Crit:** {p.crit_chance*100:.2f}% Chance | {p.crit_dmg_mult:,.2f}x Multiplier")
                st.write(f"**Super Crit:** {p.super_crit_chance*100:.2f}% Chance | {p.super_crit_dmg_mult:,.2f}x Multiplier")
                st.write(f"**Ultra Crit:** {p.ultra_crit_chance*100:.2f}% Chance | {p.ultra_crit_dmg_mult:,.2f}x Multiplier")
                
                # --- TRUE NESTED PROBABILITIES & COMPOUND MULTIPLIERS (HIDDEN) ---
                with st.expander("🔬 View Simulation Math (True Nested Probs)"):
                    true_reg = 1.0 - p.crit_chance
                    true_crit = p.crit_chance * (1.0 - p.super_crit_chance)
                    true_scrit = p.crit_chance * p.super_crit_chance * (1.0 - p.ultra_crit_chance)
                    true_ucrit = p.crit_chance * p.super_crit_chance * p.ultra_crit_chance
                    
                    comp_crit = p.crit_dmg_mult
                    comp_scrit = p.crit_dmg_mult * p.super_crit_dmg_mult
                    comp_ucrit = p.crit_dmg_mult * p.super_crit_dmg_mult * p.ultra_crit_dmg_mult
                    
                    st.write(f"*- Regular:* {true_reg*100:,.2f}%")
                    st.write(f"*- Crit:* {true_crit*100:,.2f}% *(Total Mult: {comp_crit:,.2f}x)*")
                    st.write(f"*- Super Crit:* {true_scrit*100:,.2f}% *(Total Mult: {comp_scrit:,.2f}x)*")
                    st.write(f"*- Ultra Crit:* {true_ucrit*100:,.2f}% *(Total Mult: {comp_ucrit:,.2f}x)*")

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

    # --- TAB 5: BLOCK STATS ---
    with tab_block_stats:
        st.subheader("🪨 Block Compendium")
        
        col_block_toggle, col_block_floor = st.columns([1, 1])
        with col_block_toggle:
            show_modified = st.toggle("Show Modified Stats (Applies player multipliers, cards, and floor scaling)")
        
        target_floor = 1
        if show_modified:
            with col_block_floor:
                target_floor = st.number_input("Calculate scaling for Floor Level:", min_value=1, value=int(p.current_max_floor), step=1)
                
        st.divider()

        FRAG_NAMES = {0: "Dirt", 1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Mythic", 6: "Divine"}
        table_data =[]
        
        for block_id, base in cfg.BLOCK_BASE_STATS.items():
            # Hide Tier 4 blocks if Asc2 is not unlocked
            if not p.asc2_unlocked and block_id.endswith('4'):
                continue
                
            img_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{block_id}.png")
            img_uri = get_scaled_image_uri(img_path, UI_BLOCK_TABLE_IMG_WIDTH)
            frag_name = FRAG_NAMES.get(base.get('ft', 0), "Unknown")
            
            if show_modified:
                # Feed it into the Layer 2 engine to get exact scaled math
                block_obj = Block(block_id, target_floor, p)
                
                # Apply Player Armor Penetration to the scaled armor!
                eff_armor = max(0, block_obj.armor - p.armor_pen)
                
                table_data.append({
                    "Icon": img_uri,
                    "Block": block_id.capitalize(),
                    "HP": f"{block_obj.hp:,}",
                    "Eff. Armor": f"{eff_armor:,.0f} (Base: {block_obj.armor:,})",
                    "XP Yield": f"{block_obj.xp:,.2f}",
                    "Frag Yield": f"{block_obj.frag_amt:,.3f}",
                    "Frag Type": frag_name
                })
            else:
                # Just show the raw dictionary values
                table_data.append({
                    "Icon": img_uri,
                    "Block": block_id.capitalize(),
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
                    "Icon": st.column_config.ImageColumn("Icon", help="Block Icon"),
                },
                hide_index=True,
                width="stretch",
                height=600 # Makes the table nice and tall so you don't have to scroll constantly
            )

    # --- TAB 6: HIT CALCULATOR (SANDBOX) ---
    with tab_sandbox:
        st.header("🧪 Block Hit Sandbox")
        st.write("Experiment with stat distributions without worrying about your global point budget. Find the exact breakpoints for how many hits it takes to kill specific blocks.")
        
        # --- MATH & FORMULAS (MOVED TO TOP) ---
        with st.expander("📚 Math & Formulas Breakdown (Click to expand)"):
            st.markdown("""
            **Legend:**  
            `P[x]` = Probability of x  |  `M[x]` = Multiplier of x  

            **Hit Damage Formulas:**
            * **Armor** = `(Base Armor) - (Armor Pen)`
            * **Regular Hit** = `(Damage - Armor)`
            * **Crit Hit** = `(Damage - Armor) × M[Crit]`
            * **Super Crit Hit** = `(Damage - Armor) × M[Crit] × M[sCrit]`
            * **Ultra Crit Hit** = `(Damage - Armor) × M[Crit] × M[sCrit] × M[uCrit]`
            
            **True Crit Probabilities (Nested):**
            * **P[Reg]** = `(1 - Crit Chance)`
            * **P[Crit]** = `(Crit Chance) × (1 - sCrit Chance)`
            * **P[sCrit]** = `(Crit Chance) × (sCrit Chance) × (1 - uCrit Chance)`
            * **P[uCrit]** = `(Crit Chance) × (sCrit Chance) × (uCrit Chance)`
            
            **Expected Damage Per Swing (EDPS):**  
            `EDPS = (P[Reg]×1.0 + P[Crit]×M[Crit] + P[sCrit]×M[sCrit] + P[uCrit]×M[uCrit]) × (Damage - Armor)`
            """)

        st.divider()
        
        # --- DASHBOARD LAYOUT SPLIT ---
        # Left side (1 part) for Controls, Right side (3 parts) for the Data Table
        col_controls, col_table = st.columns([1, 3])

        with col_controls:
            # Wrap the controls in an expander so the user can collapse the panel
            with st.expander("🎛️ Control Panel", expanded=True):
                
                # --- SANDBOX SYNC ---
                if st.button("🔄 Sync from Base Stats", use_container_width=True, help="Pull your currently saved stat distribution into the sandbox."):
                    for stat in STAT_CAPS.keys():
                        st.session_state[f"sandbox_stat_{stat}"] = int(p.base_stats.get(stat, 0))
                    st.rerun()

                st.markdown("#### Sandbox Stats")
                
                def render_sandbox_stat(label, stat_key, col):
                    max_val = int(STAT_CAPS[stat_key])
                    widget_key = f"sandbox_stat_{stat_key}"
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = int(p.base_stats.get(stat_key, 0))
                    
                    with col:
                        with st.container(border=True):
                            st.markdown(f"<div style='text-align: center; margin-bottom: 5px; font-size: 0.9em;'><b>{label}</b></div>", unsafe_allow_html=True)
                            
                            # --- MINI STAT ICON INJECTION ---
                            # Looks for the new small folder, falls back to the original large folder if missing
                            img_path_small = os.path.join(ROOT_DIR, "assets", "stats_small", f"{stat_key.lower()}.png")
                            img_path_large = os.path.join(ROOT_DIR, "assets", "stats", f"{stat_key.lower()}.png")
                            
                            if os.path.exists(img_path_small):
                                render_centered_image(img_path_small, 40)
                            elif os.path.exists(img_path_large):
                                render_centered_image(img_path_large, 40)
                                
                            st.number_input(
                                f"{label} Sandbox", key=widget_key, step=1, on_change=enforce_caps,
                                args=(widget_key, 0, max_val, f"Sandbox {label}"), label_visibility="collapsed"
                            )

                # Compact 2-column grid inside the left panel for the stats
                scol1, scol2 = st.columns(2)
                render_sandbox_stat("Strength", 'Str', scol1)
                render_sandbox_stat("Agility", 'Agi', scol2)
                render_sandbox_stat("Perception", 'Per', scol1)
                render_sandbox_stat("Intelligence", 'Int', scol2)
                render_sandbox_stat("Luck", 'Luck', scol1)
                render_sandbox_stat("Divine", 'Div', scol2)
                if p.asc2_unlocked:
                    render_sandbox_stat("Corruption", 'Corr', scol1)

                st.divider()
                
                st.markdown("#### Settings")
                if "sandbox_floor" not in st.session_state:
                    st.session_state["sandbox_floor"] = int(p.current_max_floor)
                    
                target_floor = st.number_input("Target Floor:", min_value=1, step=1, key="sandbox_floor")
                min_hits = st.number_input("Min Avg Hits to Kill:", min_value=1, value=1, step=1, help="Hides blocks that take fewer hits than this.")
                    
                show_unreachable = st.checkbox("Show Blocks Above Target Floor")
                show_crit_details = st.checkbox("Show Detailed Crit Multipliers")

        with col_table:
            # Build an isolated Sandbox Player Object
            import copy
            sandbox_p = copy.deepcopy(p)
            
            # Inject Sandbox Stats into the isolated clone
            for stat in STAT_CAPS.keys():
                if sandbox_p.asc2_unlocked or stat != 'Corr':
                    sandbox_p.base_stats[stat] = st.session_state.get(f"sandbox_stat_{stat}", 0)
                
            # Combat Math Extraction
            p_dmg = sandbox_p.damage
            p_enr_dmg = sandbox_p.enraged_damage
            p_pen = sandbox_p.armor_pen
            
            t_reg = 1.0 - sandbox_p.crit_chance
            t_crit = sandbox_p.crit_chance * (1.0 - sandbox_p.super_crit_chance)
            t_scrit = sandbox_p.crit_chance * sandbox_p.super_crit_chance * (1.0 - sandbox_p.ultra_crit_chance)
            t_ucrit = sandbox_p.crit_chance * sandbox_p.super_crit_chance * sandbox_p.ultra_crit_chance
            
            c_crit = sandbox_p.crit_dmg_mult
            c_scrit = c_crit * sandbox_p.super_crit_dmg_mult
            c_ucrit = c_scrit * sandbox_p.ultra_crit_dmg_mult
            
            c_enr_crit = sandbox_p.enraged_crit_dmg_mult
            c_enr_scrit = c_enr_crit * sandbox_p.super_crit_dmg_mult
            c_enr_ucrit = c_enr_scrit * sandbox_p.ultra_crit_dmg_mult
            
            avg_mult = t_reg*1.0 + t_crit*c_crit + t_scrit*c_scrit + t_ucrit*c_ucrit
            avg_enr_mult = t_reg*1.0 + t_crit*c_enr_crit + t_scrit*c_enr_scrit + t_ucrit*c_enr_ucrit
            
            # Generate the Table
            sb_table_data =[]
            for block_id in cfg.BLOCK_BASE_STATS.keys():
                tier = int(block_id[-1])
                
                if not show_unreachable:
                    if tier == 2 and target_floor <= 50: continue
                    if tier == 3 and target_floor <= 100: continue
                    if tier == 4 and target_floor <= 150: continue
                if tier == 4 and not sandbox_p.asc2_unlocked: continue
                    
                block_obj = Block(block_id, target_floor, sandbox_p)
                eff_armor = max(0, block_obj.armor - p_pen)
                
                reg_hit = max(1.0, p_dmg - eff_armor)
                enr_hit = max(1.0, p_enr_dmg - eff_armor)
                
                edps = reg_hit * avg_mult
                enr_edps = enr_hit * avg_enr_mult
                
                max_sta = math.ceil(block_obj.hp / reg_hit)
                avg_sta = math.ceil(block_obj.hp / edps)
                max_enr_sta = math.ceil(block_obj.hp / enr_hit)
                avg_enr_sta = math.ceil(block_obj.hp / enr_edps)
                
                img_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{block_id}.png")
                img_uri = get_scaled_image_uri(img_path, UI_BLOCK_TABLE_IMG_WIDTH)
                
                row = {
                    "Icon": img_uri,
                    "Block": block_id.capitalize(),
                    "HP": int(block_obj.hp),
                    "Armor": int(eff_armor),
                    "EDPS": int(edps),
                    "Enr EDPS": int(enr_edps),
                    "Reg Hit": int(reg_hit)
                }
                
                if show_crit_details:
                    row["Crit"] = int(reg_hit * c_crit)
                    row["sCrit"] = int(reg_hit * c_scrit)
                    row["uCrit"] = int(reg_hit * c_ucrit)
                    
                row["Max Hits"] = int(max_sta)
                row["Avg Hits"] = int(avg_sta)
                row["Enr Hit"] = int(enr_hit)
                
                if show_crit_details:
                    row["Enr Crit"] = int(enr_hit * c_enr_crit)
                    row["Enr sCrit"] = int(enr_hit * c_enr_scrit)
                    row["Enr uCrit"] = int(enr_hit * c_enr_ucrit)
                    
                row["Enr Max Hits"] = int(max_enr_sta)
                row["Enr Avg Hits"] = int(avg_enr_sta)
                    
                sb_table_data.append(row)
                
            if sb_table_data:
                df_sandbox = pd.DataFrame(sb_table_data)
                
                # --- EXPLICIT UI FILTERS ---
                all_blocks = df_sandbox["Block"].unique().tolist()
                selected_blocks = st.multiselect("🔍 Filter by Specific Blocks", options=all_blocks, default=[], placeholder="Select specific blocks to filter...")
                    
                if selected_blocks:
                    df_sandbox = df_sandbox[df_sandbox["Block"].isin(selected_blocks)]
                    
                if min_hits > 1:
                    df_sandbox = df_sandbox[df_sandbox["Avg Hits"] >= min_hits]

                st.markdown(f"#### 🎯 Target Breakpoints <span style='font-size: 0.6em; color: gray;'>({len(df_sandbox)} Blocks Displayed)</span>", unsafe_allow_html=True)
                
                # --- TOOLTIPS & COMMA FORMATTING ---
                num_cfg = st.column_config.NumberColumn(format="%,d")
                col_config = {
                    "Icon": st.column_config.ImageColumn("Icon", help="Block Icon"),
                    "HP": num_cfg,
                    "Armor": num_cfg,
                    "EDPS": st.column_config.NumberColumn(format="%,d", help="Expected Damage Per Swing (Average over time factoring crits)"),
                    "Enr EDPS": st.column_config.NumberColumn(format="%,d", help="Enraged Expected Damage Per Swing"),
                    "Reg Hit": num_cfg,
                    "Max Hits": st.column_config.NumberColumn(format="%,d", help="Max regular hits to kill (Worst case, no crits)"),
                    "Avg Hits": st.column_config.NumberColumn(format="%,d", help="Average regular hits to kill (Factoring crits)"),
                    "Enr Hit": st.column_config.NumberColumn(format="%,d", help="Damage of a Regular Hit while Enraged"),
                    "Enr Max Hits": st.column_config.NumberColumn(format="%,d", help="Max enraged hits to kill (Worst case, no crits)"),
                    "Enr Avg Hits": st.column_config.NumberColumn(format="%,d", help="Average enraged hits to kill (Factoring crits)")
                }
                
                if show_crit_details:
                    col_config.update({
                        "Crit": num_cfg, "sCrit": num_cfg, "uCrit": num_cfg,
                        "Enr Crit": st.column_config.NumberColumn(format="%,d", help="Enraged Critical Hit"),
                        "Enr sCrit": st.column_config.NumberColumn(format="%,d", help="Enraged Super Crit Hit"),
                        "Enr uCrit": st.column_config.NumberColumn(format="%,d", help="Enraged Ultra Crit Hit")
                    })

                st.dataframe(
                    df_sandbox,
                    column_config=col_config,
                    hide_index=True,
                    use_container_width=True,
                    height=700 # Forces the table to be nice and tall to match the left panel
                )
            
    # --- TAB 7: RUN OPTIMIZER ---
    with tab_optimizer:
        st.header("🚀 Monte Carlo Stat Optimizer")
        st.write("Leverage Successive Halving to find the absolute mathematically perfect stat distribution. Ensure your total allocated points do not exceed your budget before running.")

        # --- PROJECTION DISCLAIMER ---
        st.warning(
            "**⚠️ IMPORTANT DISCLAIMER REGARDING PROJECTIONS:**\n\n"
            "This tool is highly accurate at finding the **optimal stat distribution** for your target. "
            "However, please take the **absolute output numbers** (Max Floor, Yields/hr, Drop Times) with a grain of salt. "
            "Extensive testing shows these projections lean conservative and will likely fall short of your actual in-game "
            "performance (typically by 5-10 floors for late-game Asc1 players).\n\n"
            "*Why?* While the combat math is exact, the enemy spawn data for Floors 100+ is based on a limited sample "
            "size of real-world runs. Treat the output yields as a baseline, not absolute gospel!"
        )

        # --- GOAL SELECTION ---
        col_goal, col_target = st.columns(2)
        with col_goal:
            opt_goal = st.selectbox(
                "Optimization Target", ["Max Floor Push", "Max EXP Yield", "Fragment Farming", "Block Card Farming"]
            )
        
        with col_target:
            target_metric = "highest_floor" # Default fallback
            if opt_goal == "Fragment Farming":
                frag_tier = st.selectbox(
                    "Fragment Tier",[0, 1, 2, 3, 4, 5, 6], 
                    format_func=lambda x: {0:"Dirt", 1:"Common", 2:"Rare", 3:"Epic", 4:"Legendary", 5:"Mythic", 6:"Divine"}.get(x)
                )
                target_metric = f"frag_{frag_tier}_per_min"
            elif opt_goal == "Block Card Farming":
                block_target = st.text_input("Target Block ID (e.g., com1, myth3)", value="myth3").lower()
                target_metric = f"block_{block_target}_per_min"
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
                    STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                    if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                    
                    # Create the In-Memory Dictionary Snapshot
                    base_state_dict = {
                    'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                    'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                    'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                    'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                    'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus
                }
                    
                payload = {'stats': {s: int(p.base_stats.get(s, 0)) for s in STATS_TO_OPTIMIZE}, 'fixed_stats': {}, 'state_file': temp_run_file}
                
                # Cloud OOM Protection: Streamlit Linux containers only have 1GB RAM
                if sys.platform == "linux":
                    CPU_CORES = min(2, mp.cpu_count()) 
                else:
                    CPU_CORES = max(1, mp.cpu_count() - 1)
                    
                    with mp.Pool(CPU_CORES) as pool:
                        spd = benchmark_hardware(payload, pool)
                        st.session_state.sims_per_sec = spd
                        
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

            st.divider()
            st.write("#### 3. Execution Time Limit")
            time_limit_mins = st.slider(
                "Safely abort and return best build if time exceeds:", 
                min_value=1, max_value=30, value=5, step=1, format="%d mins"
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

    # --- MONTE CARLO EXECUTION LOOP ---
        st.divider()
        
        # Hidden for Production Beta. Change to True if you need to do UI testing later!
        # dev_mode = st.toggle("🛠️ UI Dev Mode (Instantly mock results to design UI without running engine)")
        dev_mode = False
        
        if st.button("🚀 Run Optimizer", use_container_width=True, type="primary"):
            st.write("---")
            
            # ==========================================
            # DEV MODE INTERCEPT (Instant UI Testing)
            # ==========================================
            if dev_mode:
                best_final = {'Str': 25, 'Agi': 0, 'Per': 5, 'Int': 35, 'Luck': 40, 'Div': 0}
                if p.asc2_unlocked: best_final['Corr'] = 5
                
                # Generate a mock granular trace (24 ticks per floor)
                mock_floors =[120 + (i // 24) for i in range(120)]
                mock_stamina =[max(0, 10000 - (i**1.85)) for i in range(120)]
                
                final_summary_out = {
                    target_metric: 450.5, "avg_floor": 125.4, 
                    "abs_max_floor": 132, "abs_max_chance": 0.05, "avg_metrics": {},
                    "stamina_trace": {"floor": mock_floors, "stamina": mock_stamina}
                }
                elapsed = 0.01
                time_limit_secs = 999
                
                worst_val = 45.0
                avg_val = 247.5
                runner_up_val = 432.5
                
                chart_hill_scores =[380.0, 432.5, 450.5]
                chart_hill_labels =["P1 (Coarse)", "P2 (Fine)", "P3 (Exact)"]
                chart_loot = {"Dirt": 50, "Common": 150, "Rare": 300, "Epic": 500, "Legendary": 250, "Mythic": 450, "Divine": 10}
                chart_hist = {"124": 5, "125": 12, "126": 78, "127": 5}

            # ==========================================
            # REAL ENGINE EXECUTION
            # ==========================================
            else:
                # Create the In-Memory Dictionary Snapshot for the Engine
                base_state_dict = {
                    'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                    'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                    'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                    'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                    'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus
                }

                # AUTO-BENCHMARK FAILSAFE
                if st.session_state.sims_per_sec == 0:
                    with st.spinner("⏱️ First-time setup: Benchmarking your CPU..."):
                        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                        if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                        payload = {'stats': {s: int(p.base_stats.get(s, 0)) for s in STATS_TO_OPTIMIZE}, 'fixed_stats': {}, 'state_dict': base_state_dict}
                        # Cloud OOM Protection: Streamlit Linux containers only have 1GB RAM
                        if sys.platform == "linux":
                            CPU_CORES = min(2, mp.cpu_count()) 
                        else:
                            CPU_CORES = max(1, mp.cpu_count() - 1)
                        with mp.Pool(CPU_CORES) as pool:
                            spd = benchmark_hardware(payload, pool)
                            st.session_state.sims_per_sec = spd
                            budget = int(sum(p.base_stats.get(s, 0) for s in STATS_TO_OPTIMIZE))
                            cap_increase = int(p.u('H45'))
                            caps = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in STATS_TO_OPTIMIZE}
                            st.session_state.eta_profiles = get_eta_profiles(STATS_TO_OPTIMIZE, budget, caps, spd)

                prof_key = next((k for k in st.session_state.eta_profiles.keys() if k.startswith(depth_choice)), None)
                if prof_key:
                    prof_data = st.session_state.eta_profiles[prof_key]
                    step_size = prof_data['step']
                    st.info(f"⏱️ **Running {depth_choice} Search:** Estimated to take {prof_data['time_label']} (~{prof_data['builds']:,.0f} builds at {st.session_state.sims_per_sec:,.0f} sims/sec)")

                with st.spinner(f"Engine Running..."):
                    start_time = time.time()
                    STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                    if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                    DYNAMIC_BUDGET = int(sum(st.session_state.get(f"stat_{s}", p.base_stats.get(s, 0)) for s in STATS_TO_OPTIMIZE))
                    FIXED_STATS = {k: v for k, v in p.base_stats.items() if k not in STATS_TO_OPTIMIZE}
                    cap_increase = int(p.u('H45'))
                    EFFECTIVE_CAPS = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in STATS_TO_OPTIMIZE}
                    
                    # Cloud OOM Protection: Streamlit Linux containers only have 1GB RAM
                    if sys.platform == "linux":
                        CPU_CORES = min(2, mp.cpu_count()) 
                    else:
                        CPU_CORES = max(1, mp.cpu_count() - 1)
                    ITER_P1, ITER_P2, ITER_P3 = 25, 50, 100
                    best_p3, final_summary = None, None
                    
                    ui_prog_bar = st.progress(0, text="Booting up engine cores...")
                    def st_progress_callback(phase_name, r_idx, r_total, task_idx, task_total):
                        pct = min(100, max(0, int((task_idx / task_total) * 100)))
                        ui_prog_bar.progress(pct, text=f"⚙️ {phase_name} | Round {r_idx}/{r_total} | {pct}% ({task_idx}/{task_total} sims completed)")
                    
                    time_limit_secs = time_limit_mins * 60
                    
                    with mp.Pool(CPU_CORES) as pool:
                        bounds_p1 = {s: (0, EFFECTIVE_CAPS[s]) for s in STATS_TO_OPTIMIZE}
                        best_p1, summary_p1 = run_optimization_phase(
                            "Phase 1 (Coarse)", target_metric, STATS_TO_OPTIMIZE, 
                            DYNAMIC_BUDGET, step_size, ITER_P1, pool, FIXED_STATS, bounds_p1,
                            progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                            base_state_dict=base_state_dict # <--- Passed via memory!
                        )
                        
                        best_p2, summary_p2 = None, None
                        if best_p1 and (time.time() - start_time) < time_limit_secs:
                            bounds_p2 = {s: (max(0, best_p1[s] - step_size), min(EFFECTIVE_CAPS[s], best_p1[s] + step_size)) for s in STATS_TO_OPTIMIZE}
                            step_2 = max(2, step_size // 3)
                            best_p2, summary_p2 = run_optimization_phase(
                                "Phase 2 (Fine)", target_metric, STATS_TO_OPTIMIZE, 
                                DYNAMIC_BUDGET, step_2, ITER_P2, pool, FIXED_STATS, bounds_p2,
                                progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                                base_state_dict=base_state_dict
                            )
                            
                        if best_p2 and (time.time() - start_time) < time_limit_secs:
                            p3_radius = min(2, step_2) 
                            bounds_p3 = {s: (max(0, best_p2[s] - p3_radius), min(EFFECTIVE_CAPS[s], best_p2[s] + p3_radius)) for s in STATS_TO_OPTIMIZE}
                            best_p3, final_summary = run_optimization_phase(
                                "Phase 3 (Exact)", target_metric, STATS_TO_OPTIMIZE, 
                                DYNAMIC_BUDGET, 1, ITER_P3, pool, FIXED_STATS, bounds_p3,
                                progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                                base_state_dict=base_state_dict
                            )
                    
                    best_final = best_p3 or best_p2 or best_p1
                    final_summary_out = final_summary or summary_p2 or summary_p1
                    ui_prog_bar.empty()
                    elapsed = time.time() - start_time
                    
                    # --- MAP REAL TELEMETRY ---
                    if final_summary_out:
                        worst_val = final_summary_out.get("worst_val", 0)
                        avg_val = final_summary_out.get("avg_val", 0)
                        runner_up_val = final_summary_out.get("runner_up_val", 0)
                        
                        chart_hill_scores =[]
                        chart_hill_labels =[]
                        if summary_p1: chart_hill_scores.append(summary_p1[target_metric]); chart_hill_labels.append("P1 (Coarse)")
                        if summary_p2: chart_hill_scores.append(summary_p2[target_metric]); chart_hill_labels.append("P2 (Fine)")
                        if final_summary: chart_hill_scores.append(final_summary[target_metric]); chart_hill_labels.append("P3 (Exact)")
                        
                        floors = final_summary_out.get("floors",[])
                        chart_hist = dict(Counter(floors))
                        
                        chart_loot = {}
                        frag_names = {0:"Dirt", 1:"Common", 2:"Rare", 3:"Epic", 4:"Legendary", 5:"Mythic", 6:"Divine"}
                        avg_metrics = final_summary_out.get("avg_metrics", {})
                        for tier, name in frag_names.items():
                            k = f"frag_{tier}_per_min"
                            val = avg_metrics.get(k, 0)
                            if val > 0:
                                chart_loot[name] = val

            # ==========================================
            # UI RESULTS TELEMETRY (DYNAMIC RENDER)
            # ==========================================
            if best_final and final_summary_out:
                if elapsed >= time_limit_secs:
                    st.warning(f"⚠️ **Time Limit Reached!** Optimization safely aborted early at {elapsed:.1f} seconds. Showing the best build found so far.")
                else:
                    st.success(f"✅ Successive Halving Complete in {elapsed:.1f} seconds!")
                
                st.markdown("### 🏆 Optimal Stat Build")
                st.write("*(Green/Red numbers show changes from your current UI allocation)*")
                
                stat_cols = st.columns(len(best_final))
                for idx, (stat_name, allocated_pts) in enumerate(best_final.items()):
                    with stat_cols[idx]:
                        with st.container(border=True):
                            img_path = os.path.join(ROOT_DIR, "assets", "stats", f"{stat_name.lower()}.png")
                            if os.path.exists(img_path):
                                render_centered_image(img_path, 250) 
                            else:
                                st.markdown(f"<div style='text-align:center;'><b>{stat_name}</b></div>", unsafe_allow_html=True)
                            
                            current_val = int(st.session_state.get(f"stat_{stat_name}", p.base_stats.get(stat_name, 0)))
                            delta = int(allocated_pts) - current_val
                            st.metric(label=stat_name, value=int(allocated_pts), delta=delta, label_visibility="collapsed")
                
                if st.button("✨ Apply Build to UI", use_container_width=True):
                    for k, v in best_final.items():
                        st.session_state[f"stat_{k}"] = int(v)
                        p.base_stats[k] = int(v)
                    st.rerun()

                st.divider()

                # ==========================================
                # ADVANCED ANALYTICS DASHBOARD (TABS)
                # ==========================================
                st.markdown("### 📊 Advanced Analytics Dashboard")
                tab_list =["📈 Performance"]
                # Show Loot for any farming/EXP mode, Show Wall strictly for Floor Pushing
                show_loot = (target_metric != "highest_floor" or dev_mode)
                show_wall = (target_metric == "highest_floor" or dev_mode)
                
                if show_loot: tab_list.append("🎒 Loot Breakdown")
                if show_wall: tab_list.append("🧱 The Wall")
                
                ui_tabs = st.tabs(tab_list)
                tab_idx = 0
                
                # --- TAB 1: PERFORMANCE & CONFIDENCE ---
                with ui_tabs[tab_idx]:
                    tab_idx += 1
                    perf_col1, perf_col2 = st.columns([1, 1.5])
                    
                    with perf_col1:
                        if target_metric == "highest_floor":
                            st.markdown("#### 🏆 Push Potential")
                            
                            # Use the specific telemetry calculated in parallel_worker.py
                            abs_max = final_summary_out.get("abs_max_floor", final_summary_out.get(target_metric, 0))
                            abs_chance = final_summary_out.get("abs_max_chance", 0) * 100
                            avg_flr = final_summary_out.get("avg_floor", final_summary_out.get(target_metric, 0))
                            
                            st.metric("Absolute Max Floor (God Run)", f"Floor {abs_max:,.0f}")
                            st.metric("God Run Probability", f"{abs_chance:.1f}%")
                            st.metric("Average Consistency Floor", f"Floor {avg_flr:,.1f}")
                        else:
                            val = final_summary_out[target_metric]
                            rate_1k = (val / 60.0) * 1000.0
                            metric_str = "Fragments" if "frag" in target_metric else "Kills" if "block" in target_metric else "EXP"
                            
                            # Clean, consolidated Banked Time readout
                            st.markdown(f"#### 💰 Projected Yield<br><span style='font-size: 0.9em; color: gray;'>Target {metric_str} per 1k Arch Seconds</span>", unsafe_allow_html=True)
                            st.metric("Yield", f"{rate_1k:,.1f}", label_visibility="collapsed")
                            
                            st.divider()
                            
                            # Clean, consolidated Real-Time readout
                            # Clean, consolidated Real-Time readout
                            st.markdown(f"#### ⏱️ Real-Time Yield<br><span style='font-size: 0.9em; color: gray;'>{metric_str} / minute</span>", unsafe_allow_html=True)
                            st.metric("Real-Time", f"{val:,.2f}", label_visibility="collapsed")
                            
                            # --- NEW BLOCK CARD ODDS SECTION ---
                            if "block_" in target_metric:
                                st.divider()
                                st.markdown(f"#### 🃏 Block Card Drop Estimates<br><span style='font-size: 0.9em; color: gray;'>Based on {val:,.2f} target kills/min</span>", unsafe_allow_html=True)
                                
                                odds = {"Base Card": 1500, "Poly Fragments": 7500, "Infernal Fragments": 200000}
                                
                                # Extract the exact core image needed (e.g., "block_myth3_per_min" -> "myth3")
                                target_block_id = target_metric.replace("block_", "").replace("_per_min", "")
                                cblock_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{target_block_id}.png")
                                
                                # Map backgrounds: Base=1(Com), Poly=2(Rare), Infernal=4(Leg)
                                bg_mapping = {"Base Card": "1", "Poly Fragments": "2", "Infernal Fragments": "4"}
                                
                                cols_cards = st.columns(3)
                                for idx, (drop_name, base_odds) in enumerate(odds.items()):
                                    with cols_cards[idx]:
                                        with st.container(border=True):
                                            # --- RENDER DYNAMIC CARD ---
                                            bg_tier = bg_mapping.get(drop_name, "1")
                                            bg_path = os.path.join(ROOT_DIR, "assets", "cards", "backgrounds", f"{bg_tier}.png")
                                            
                                            comp_img = composite_card(bg_path, cblock_path, UI_BLOCK_CARD_X_OFFSET, UI_BLOCK_CARD_Y_OFFSET)
                                            if comp_img:
                                                render_centered_image(comp_img, UI_BLOCK_CARD_WIDTH)
                                            else:
                                                st.markdown("<div style='text-align: center; color: gray;'><small>(Assets Missing)</small></div>", unsafe_allow_html=True)
                                            
                                            st.markdown(f"<div style='text-align: center; margin-top: -10px;'><b>{drop_name}</b><br><span style='font-size: 0.8em; color: gray;'>(1 in {base_odds:,})</span></div>", unsafe_allow_html=True)
                                            st.divider()
                                            
                                            # --- MATH & YIELDS ---
                                            if val > 0:
                                                kills_50 = 0.693 * base_odds
                                                kills_90 = 2.302 * base_odds
                                                
                                                def format_time(req_kills):
                                                    rt_mins = req_kills / val
                                                    rt_str = f"{rt_mins:.1f}m" if rt_mins < 60 else f"{rt_mins/60.0:.1f}h"
                                                    arch_secs = req_kills / (val / 60.0)
                                                    arch_1k = arch_secs / 1000.0
                                                    return rt_str, arch_1k

                                                rt_50, bk_50 = format_time(kills_50)
                                                rt_90, bk_90 = format_time(kills_90)
                                                
                                                st.markdown(f"<small><b>50% Chance (Lucky):</b><br>~{rt_50} | ~{bk_50:.1f}k Banked</small>", unsafe_allow_html=True)
                                                st.markdown(f"<small><b>90% Chance (Safe):</b><br>~{rt_90} | ~{bk_90:.1f}k Banked</small>", unsafe_allow_html=True)
                                            else:
                                                st.markdown("<div style='text-align: center; color: gray;'><small>N/A (0 kills)</small></div>", unsafe_allow_html=True)

                            st.divider()
                            
                            avg_flr = final_summary_out.get("avg_floor", 0)
                            st.markdown(f"#### 🧱 Average Death<br><span style='font-size: 0.9em; color: gray;'>Floor reached per run</span>", unsafe_allow_html=True)
                            st.metric("Avg Floor", f"Floor {avg_flr:,.1f}", label_visibility="collapsed")

                    with perf_col2:
                        # Streamlit Markdown header completely fixes the Plotly overlap bug
                        st.markdown(
                            "#### AI Convergence (Hill Climb) "
                            "<span title='This chart shows how the AI narrowed down the best build across the 3 optimization phases. "
                            "An upward curve means the engine successfully found significantly better builds as it zoomed in. "
                            "A flat line means Phase 1 already hit the near-perfect build.' "
                            "style='cursor: help; font-size: 0.8em;'>ℹ️</span>", 
                            unsafe_allow_html=True
                        )
                        df_hill = pd.DataFrame({"Phase": chart_hill_labels, "Score": chart_hill_scores})
                        fig_hill = px.line(df_hill, x="Phase", y="Score", markers=True)
                        fig_hill.update_traces(line_color='#4CAF50', marker=dict(size=10))
                        fig_hill.update_layout(margin=dict(l=10, r=20, t=10, b=20), height=200)
                        st.plotly_chart(fig_hill, use_container_width=True)
                        
                        # Streamlit Markdown header
                        st.markdown(
                            "#### Engine Confidence Analysis "
                            "<span title='Compares the Optimal build against the Worst, Average, and Runner-Up builds tested. "
                            "A large gap between Optimal and Average proves your stats highly impact this target. A small gap between Runner-Up and Optimal "
                            "shows the AI fine-tuned the absolute perfect micro-adjustments.' "
                            "style='cursor: help; font-size: 0.8em;'>ℹ️</span>", 
                            unsafe_allow_html=True
                        )
                        df_conf = pd.DataFrame({
                            "Build Category":["Worst Tested", "Average", "Runner-Up", "🏆 Optimal"],
                            "Performance":[worst_val, avg_val, runner_up_val, final_summary_out[target_metric]]
                        })
                        fig_conf = px.bar(
                            df_conf, x="Performance", y="Build Category", orientation='h', text_auto='.3s', color="Build Category",
                            color_discrete_map={"Worst Tested": "#ff4b4b", "Average": "#ffa229", "Runner-Up": "#6495ED", "🏆 Optimal": "#4CAF50"}
                        )
                        fig_conf.update_layout(showlegend=False, margin=dict(l=10, r=20, t=10, b=20), height=200)
                        st.plotly_chart(fig_conf, use_container_width=True)
                # --- TAB 2: COLLATERAL LOOT (BAR CHART) ---
                if show_loot:
                    with ui_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("#### Collateral Loot Distribution")
                        st.write("On average, each minute of simulated combat yields the following collateral rewards alongside your target:")
                        
                        total_loot = sum(chart_loot.values()) if chart_loot else 1
                        df_loot = pd.DataFrame(list(chart_loot.items()), columns=['Loot Tier', 'Amount'])
                        df_loot['Label'] = df_loot['Amount'].apply(lambda x: f"{x:,.1f}  ({(x/total_loot)*100:.1f}%)")
                        
                        fig_loot = px.bar(
                            df_loot, x='Loot Tier', y='Amount', text='Label', color='Loot Tier',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_loot.update_traces(textposition='outside')
                        fig_loot.update_layout(showlegend=False, margin=dict(t=20, b=20), height=400)
                        st.plotly_chart(fig_loot, use_container_width=True)

                # --- TAB 3: THE WALL (HISTOGRAM) ---
                if show_wall:
                    with ui_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("#### Death Distribution (The Brick Wall)")
                        st.write("Out of the simulations run on the optimal build, this is exactly where your character died. High spikes indicate a hard scaling wall (usually enemy armor).")
                        
                        df_hist = pd.DataFrame(list(chart_hist.items()), columns=['Floor', 'Deaths'])
                        # Sort the dataframe by Floor numerically so the x-axis reads chronologically
                        df_hist['Floor'] = pd.to_numeric(df_hist['Floor'])
                        df_hist = df_hist.sort_values(by='Floor')
                        
                        fig_hist = px.bar(df_hist, x='Floor', y='Deaths', text='Deaths')
                        fig_hist.update_traces(marker_color='#ff4b4b', textposition='outside')
                        fig_hist.update_layout(margin=dict(t=20, b=20), height=400, xaxis_type='category')
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # --- STAMINA PLOT (NEW) ---
                        if "stamina_trace" in final_summary_out:
                            st.divider()
                            st.markdown("#### Stamina Depletion Trace (Sample Run)")
                            st.write("A simulated look at how your stamina drains floor-by-floor. Hover over the line to see your exact remaining stamina at the end of each floor.")
                            
                            # We still receive the granular arrays from the engine...
                            trace_floors = final_summary_out["stamina_trace"]["floor"]
                            trace_stamina = final_summary_out["stamina_trace"]["stamina"]
                            
                            df_stam = pd.DataFrame({
                                "Floor": trace_floors,
                                "Stamina": trace_stamina
                            })
                            
                            # ...but we use Pandas to extract ONLY the final stamina value for each floor!
                            # This guarantees a strictly ascending X-axis and completely fixes the diagonal line bugs.
                            df_grouped = df_stam.groupby("Floor", as_index=False).last()
                            
                            fig_stam = px.line(
                                df_grouped, 
                                x="Floor", 
                                y="Stamina",
                                hover_data={"Floor": True, "Stamina": ":,.0f"}
                            )
                            # Fill area under the curve
                            fig_stam.update_traces(line_color="#ffa229", fill='tozeroy', fillcolor="rgba(255, 162, 41, 0.2)")
                            fig_stam.update_layout(
                                margin=dict(t=20, b=20), 
                                height=300,
                                xaxis_title="Floor Level"
                            )
                            st.plotly_chart(fig_stam, use_container_width=True)

            else:
                st.error("Optimization failed or aborted before a single build could be tested.")