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
from optimizers.parallel_worker import run_optimization_phase, benchmark_hardware, get_optimal_step_profile, worker_simulate

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

@st.cache_data(show_spinner=False)
def _get_base64_image(img_path, target_width):
    """Cached internal helper for rendering static path-based images."""
    if not os.path.exists(img_path): return None
    img = Image.open(img_path).convert("RGBA")
    w_percent = (target_width / float(img.width))
    target_height = int((float(img.height) * float(w_percent)))
    img_resized = img.resize((target_width, target_height), Image.NEAREST)
    buffered = BytesIO()
    img_resized.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def render_centered_image(img_source, target_width):
    """
    Physically resizes the image using PIL Nearest Neighbor for razor-sharp 
    retro pixels, completely bypassing Streamlit CSS rendering bugs.
    """
    encoded = None
    if isinstance(img_source, str):
        # Cache hits for standard file paths
        encoded = _get_base64_image(img_source, target_width)
    else:
        # Dynamic processing for composited RAM images (handled by composite_card cache)
        img = img_source
        w_percent = (target_width / float(img.width))
        target_height = int((float(img.height) * float(w_percent)))
        img_resized = img.resize((target_width, target_height), Image.NEAREST)
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode()
        
    if encoded:
        html = f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{encoded}">
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def composite_card(bg_path, cblock_path, x_offset, y_offset):
    """Dynamically overlays ANY core asset onto a dynamic background."""
    try:
        bg = Image.open(bg_path).convert("RGBA")
        
        # --- NEW: Standardize background sizes against 1.png to prevent UI jitter ---
        base_bg_path = os.path.join(ROOT_DIR, "assets", "cards", "backgrounds", "1.png")
        if os.path.exists(base_bg_path):
            with Image.open(base_bg_path) as base_bg:
                if bg.size != base_bg.size:
                    bg = bg.resize(base_bg.size, Image.NEAREST)
                    
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

@st.cache_data(show_spinner=False)
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
    else:
        # Graceful Schema Migration Failsafe: 
        # Instead of wiping their entire memory on a hot-reload, we surgically inject missing attributes.
        dummy_player = Player()
        for attr, default_val in dummy_player.__dict__.items():
            if not hasattr(st.session_state.player, attr):
                setattr(st.session_state.player, attr, default_val)

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
        
        /* Tighten the internal padding of the bordered stat containers */[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.75rem !important;
        }
        
        /* 5. Make the Number Input a CSS Container for responsive centering */[data-testid="stNumberInput"] {
            container-type: inline-size;
        }
        
        /* 6. Strip default padding and center the text inside the input area */[data-testid="stNumberInput"] input {
            text-align: center !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
            margin: 0 !important;
        }
        
        /* 7. When wide enough for +/- buttons (~48px right), pad the left by 48px to find the true center! */
        @container (min-width: 120px) {
            [data-testid="stNumberInput"] input {
                padding-left: 60px !important;
            }
        }
        
        /* 8. Floating Back to Top Button */
        .back-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background-color: #ffa229;
            color: #2b2b2b !important;
            padding: 10px 16px;
            border-radius: 20px;
            text-decoration: none !important;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.4);
            z-index: 99999;
            transition: all 0.2s ease-in-out;
            opacity: 0.8;
        }
        .back-to-top:hover {
            background-color: #2b2b2b;
            color: #ffa229 !important;
            border: 1px solid #ffa229;
            transform: translateY(-2px);
            opacity: 1.0;
        }
        </style>
    """, unsafe_allow_html=True)

    # ==========================================
    # 🔐 BETA ACCESS GATE (WITH URL MEMORY)
    # ==========================================
    # Securely grab the password from .streamlit/secrets.toml (locally) or the Streamlit Dashboard
    try:
        CORRECT_KEY = st.secrets["BETA_KEY"]
    except Exception as e:
        st.error(f"Missing secrets configuration! App locked. (Error: {e})")
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
    # MAIN WINDOW: Tabs & Navigation
    # ==========================================
    st.markdown('<div id="top-of-tabs"></div>', unsafe_allow_html=True)
    st.title("⛏️ AI Arch Mining Optimizer")

    # Calculate dynamic Base Stat caps (Base + Upgrade #45)
    cap_inc = int(p.u('H45'))
    STAT_CAPS = {
        'Str': 50 + cap_inc, 'Agi': 50 + cap_inc,
        'Per': 25 + cap_inc, 'Int': 25 + cap_inc, 'Luck': 25 + cap_inc,
        'Div': 10 + cap_inc, 'Corr': 10 + cap_inc
    }

    tab_welcome, tab_setup, tab_calc_stats, tab_block_stats, tab_sims = st.tabs([
        "🏠 Welcome", "⚙️ Player Setup", "📋 Calculated Stats", "🪨 Block Compendium", "🧪 Simulations"
    ])

    # Pre-define the Simulation sub-tabs so we can seamlessly route content to them later
    with tab_sims:
        tab_optimizer, tab_synth, tab_sandbox = st.tabs(["🚀 Optimizer", "🧬 Build Synthesis & History", "🧪 Hit Calculator (Sandbox)"])

    # ==========================================
    # PLAYER SETUP & SIDEBAR MIGRATION
    # ==========================================
    with tab_setup:
        col_setup_menu, col_setup_content = st.columns([1, 3])
        
        with col_setup_content:
            tab_stats, tab_upgrades, tab_cards, tab_idols = st.tabs(["📊 Base Stats", "⬆️ Upgrades", "🎴 Block Cards", "🗿 Arch Idols"])

        with col_setup_menu:
        
            # --- 1. GLOBAL SETTINGS ---
            with st.expander("⚙️ Global Settings", expanded=True):
                # Initialize session state for global settings so they don't throw warnings
                if "set_asc1" not in st.session_state: st.session_state["set_asc1"] = p.asc1_unlocked
                if "set_asc2" not in st.session_state: st.session_state["set_asc2"] = p.asc2_unlocked
                if "set_arch" not in st.session_state: st.session_state["set_arch"] = int(p.arch_level)
                if "set_floor" not in st.session_state: st.session_state["set_floor"] = int(p.current_max_floor)
                if "set_hades" not in st.session_state: st.session_state["set_hades"] = int(p.hades_idol_level)
                
                # Render widgets with explicit keys
                p.asc1_unlocked = st.checkbox("Ascension 1 Unlocked", key="set_asc1")
                
                # Force Asc2 to uncheck if Asc1 is disabled
                if not p.asc1_unlocked:
                    st.session_state["set_asc2"] = False
                    p.asc2_unlocked = False
                    
                p.asc2_unlocked = st.checkbox("Ascension 2 Unlocked", key="set_asc2", disabled=not p.asc1_unlocked)
                p.arch_level = st.number_input("Arch Level", min_value=1, step=1, key="set_arch")
                p.current_max_floor = st.number_input("Max Floor Reached", min_value=1, step=1, key="set_floor")

                # --- OVER-BUDGET AUTO-FIX ---
                current_allowed = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
                current_allocated = sum(int(st.session_state.get(f"stat_{s}", p.base_stats.get(s, 0))) for s in p.base_stats.keys())
                if current_allocated > current_allowed:
                    st.error(f"⚠️ **Over Budget!**\nYou have allocated {current_allocated} / {current_allowed} points.")
                    def trim_stats():
                        excess = current_allocated - current_allowed
                        # Trim stats in reverse priority order to preserve survival/damage
                        trim_order =['Corr', 'Div', 'Luck', 'Int', 'Per', 'Agi', 'Str']
                        for s in trim_order:
                            if excess <= 0: break
                            val = int(st.session_state.get(f"stat_{s}", p.base_stats.get(s, 0)))
                            if val > 0:
                                deduct = min(val, excess)
                                st.session_state[f"stat_{s}"] = val - deduct
                                p.base_stats[s] = val - deduct
                                excess -= deduct
                        st.toast("✅ Stats trimmed to fit new budget!", icon="✂️")
                    st.button("✂️ Auto-Trim Stats", width="stretch", on_click=trim_stats, type="primary")

            # --- 2. IMPORT DATA ---
            with st.expander("📂 Import Data", expanded=True):
                uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
                
                if uploaded_file is not None:
                    # Prevent infinite reloading: Only process if it is a NEW file upload!
                    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.file_id:
                        st.session_state.last_uploaded_file = uploaded_file.file_id
                        
                        import uuid
                        temp_path = os.path.join(ROOT_DIR, f"temp_upload_{uuid.uuid4().hex}.json")
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        load_state_from_json(p, temp_path)
                        if os.path.exists(temp_path):
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

                import uuid
                temp_export = os.path.join(ROOT_DIR, f"temp_export_{uuid.uuid4().hex}.json")
                save_state_to_json(p, temp_export, readable_keys=True, hide_locked=True)
                
                # --- INTERCEPT AND ENFORCE STRICT CARD JSON ORDERING ---
                with open(temp_export, "r") as f:
                    export_data = json.load(f)
                    
                # Strip hardcoded constants to prevent user confusion and math tampering
                if "settings" in export_data and "base_damage_const" in export_data["settings"]:
                    del export_data["settings"]["base_damage_const"]
                    
                if "cards" in export_data:
                    ordered_cards = {}
                    for ot in['dirt', 'com', 'rare', 'epic', 'leg', 'myth', 'div']:
                        for tier in range(1, 5):
                            cid = f"{ot}{tier}"
                            if cid in export_data["cards"]:
                                ordered_cards[cid] = export_data["cards"][cid]
                    export_data["cards"] = ordered_cards
                    
                # json.dumps without sort_keys preserves our forced insertion order
                export_json_str = json.dumps(export_data, indent=4)
                
                if os.path.exists(temp_export):
                    os.remove(temp_export)
                    
                st.download_button(
                        label="📥 Download JSON",
                        data=export_json_str,
                        file_name="player_state.json",
                        mime="application/json",
                        width="stretch"
                    )

    # --- TAB 0: WELCOME ---
    with tab_welcome:
        with st.container(border=True):
            st.markdown("### 👋 Welcome to the Optimizer!")
            st.write("If you are new here, follow these 3 steps to get started:")
            st.markdown("1. **Input your Stats & Upgrades:** Go to the **Player Setup** tab to manually enter your player info or **Import** your own json player data, or click a **Preset Build** below to auto-fill realistic data.\n2. **Select your Goal:** Go to the **Simulations -> Optimizer** tab and choose your target.\n3. **Run the Engine:** Let the Monte Carlo simulations find your perfect mathematical build.")
            
            st.divider()
            st.markdown("#### 🚀 Quick Start: Load a Preset Build")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            
            def apply_preset(preset_dict=None, reset=False):
                if reset:
                    st.session_state.player = Player()
                else:
                    import uuid
                    temp_path = os.path.join(ROOT_DIR, f"temp_preset_{uuid.uuid4().hex}.json")
                    with open(temp_path, "w") as f:
                        json.dump(preset_dict, f)
                    load_state_from_json(st.session_state.player, temp_path)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Flush all UI widget keys to force sync with the new player state
                for k in list(st.session_state.keys()):
                    if k.startswith(("upg_", "stat_", "ext_", "card_", "set_", "sandbox_")):
                        del st.session_state[k]
                st.rerun()

            with col_p1:
                if st.button("🌱 Load Early-Game Build\n(Asc 1, Floor 40)", width="stretch"):
                    early_game = {"settings": {"asc2_unlocked": False, "arch_level": 45, "current_max_floor": 40, "base_damage_const": 10, "total_infernal_cards": 0}, "base_stats": {"Str": 15, "Agi": 0, "Per": 0, "Int": 0, "Luck": 20, "Div": 10}, "internal_upgrades": {"3 - Gem Stamina": 25, "4 - Gem Exp": 12, "5 - Gem Loot": 12, "9 - Flat Damage": 15, "10 - Armor Pen.": 15, "11 - Exp. Gain": 15, "12 - Stat Points": 3, "13 - Crit Chance/Damage": 12, "14 - Max Sta/Sta Mod Chance": 12, "15 - Flat Damage": 8, "16 - Loot Mod Gain": 6, "17 - Unlock Fairy/Armor Pen": 6, "18 - Enrage&Crit Dmg/Enrage Cooldown": 5, "20 - Flat Dmg/Super Crit Chance": 5, "21 - Exp Gain/Fragment Gain": 4, "22 - Flurry Sta Gain/Flurr Cooldown": 4, "23 - Max Sta/Sta Mod Gain": 4, "24 - All Mod Chances": 3, "25 - Flat Dmg/Damage Up": 0, "26 - Max Sta/Mod Chance": 0, "28 - Exp Gain/Max Sta": 3, "29 - Armor Pen/Ability Cooldowns": 3, "30 - Crit Dmg/Super Crit Dmg": 3, "31 - Quake Atks/Cooldown": 3, "32 - Flat Dmg/Enrage Cooldown": 0, "33 - Mod Chance/Armor Pen": 0, "35 - Exp Gain/Mod Ch.": 0, "36 - Damage Up/Armor Pen": 0, "37 - Super Crit/Ultra Crit Chance": 0, "38 - Exp Mod Gain/Chance": 0, "39 - Ability Insta Chance/Max Sta": 0, "40 - Ultra Crit Dmg/Sta Mod Chance": 0, "41 - Poly Card Bonus": 0, "42 - Frag Gain Mult": 0, "43 - Sta Mod Gain": 0, "44 - All Mod Chances": 0, "45 - Exp Gain/All Stat Cap Inc.": 0, "47 - Damage Up/Crit Dmg Up": 0, "48 - Gold Crosshair Chance/Auto-Tap Chance": 0, "49 - Flat Dmg/Ultra Crit Chance": 0, "50 - Ability Insta Chance/Sta Mod Chance": 0, "51 - Dmg Up/Exp Gain": 0, "53 - Super Crit Dmg/Exp Mod Gain": 0, "54 - Max Sta/Crosshair Auto-Tap Chance": 0}, "external_upgrades": {"Hestia Idol": 0, "Axolotl Skin": 9, "Dino Skin": 9, "Geoduck Tribute": 750, "Avada Keda- Skill": 1, "Block Bonker Skill": 1, "Archaeology Bundle": 0, "Ascension Bundle": 0, "Arch Ability Card": 3, "Arch Ability Infernal Bonus": 0.0}, "cards": {"dirt1": 3, "dirt2": 2, "dirt3": 2, "com1": 3, "com2": 2, "com3": 2, "rare1": 3, "rare2": 2, "rare3": 2, "epic1": 2, "epic2": 2, "epic3": 2, "leg1": 2, "leg2": 2, "leg3": 2, "myth1": 2, "myth2": 2, "myth3": 2, "div1": 2, "div2": 0, "div3": 0}}
                    apply_preset(early_game)
                    
            with col_p2:
                if st.button("🌌 Load Late-Game Build\n(Asc 2, Floor 158)", width="stretch"):
                    late_game = {"settings": {"asc2_unlocked": True, "arch_level": 99, "current_max_floor": 158, "base_damage_const": 10, "hades_idol_level": 129, "total_infernal_cards": 303}, "base_stats": {"Str": 15, "Agi": 0, "Per": 0, "Int": 29, "Luck": 30, "Div": 15, "Corr": 15}, "internal_upgrades": {"3 - Gem Stamina": 50, "4 - Gem Exp": 25, "5 - Gem Loot": 25, "9 - Flat Damage": 25, "10 - Armor Pen.": 25, "11 - Exp. Gain": 25, "12 - Stat Points": 5, "13 - Crit Chance/Damage": 25, "14 - Max Sta/Sta Mod Chance": 20, "15 - Flat Damage": 20, "16 - Loot Mod Gain": 10, "17 - Unlock Fairy/Armor Pen": 15, "18 - Enrage&Crit Dmg/Enrage Cooldown": 15, "19 - Gleaming Floor Chance": 30, "20 - Flat Dmg/Super Crit Chance": 25, "21 - Exp Gain/Fragment Gain": 20, "22 - Flurry Sta Gain/Flurr Cooldown": 10, "23 - Max Sta/Sta Mod Gain": 5, "24 - All Mod Chances": 30, "25 - Flat Dmg/Damage Up": 5, "26 - Max Sta/Mod Chance": 5, "27 - Unlock Ability Fairy/Loot Mod Gain": 20, "28 - Exp Gain/Max Sta": 15, "29 - Armor Pen/Ability Cooldowns": 10, "30 - Crit Dmg/Super Crit Dmg": 20, "31 - Quake Atks/Cooldown": 10, "32 - Flat Dmg/Enrage Cooldown": 5, "33 - Mod Chance/Armor Pen": 5, "34 - Buff Divinity[Div Stats Up]": 5, "35 - Exp Gain/Mod Ch.": 5, "36 - Damage Up/Armor Pen": 20, "37 - Super Crit/Ultra Crit Chance": 20, "38 - Exp Mod Gain/Chance": 20, "39 - Ability Insta Chance/Max Sta": 20, "40 - Ultra Crit Dmg/Sta Mod Chance": 20, "41 - Poly Card Bonus": 1, "42 - Frag Gain Mult": 1, "43 - Sta Mod Gain": 1, "44 - All Mod Chances": 1, "45 - Exp Gain/All Stat Cap Inc.": 1, "46 - Gleaming Floor Multi": 24, "47 - Damage Up/Crit Dmg Up": 1, "48 - Gold Crosshair Chance/Auto-Tap Chance": 5, "49 - Flat Dmg/Ultra Crit Chance": 5, "50 - Ability Insta Chance/Sta Mod Chance": 25, "51 - Dmg Up/Exp Gain": 5, "52 - [Corruption Buff] Dmg Up / Mod Multi Up": 10, "53 - Super Crit Dmg/Exp Mod Gain": 30, "54 - Max Sta/Crosshair Auto-Tap Chance": 28, "55 - All Mod Multipliers": 10}, "external_upgrades": {"Hestia Idol": 1929, "Axolotl Skin": 11, "Dino Skin": 11, "Geoduck Tribute": 1047, "Avada Keda- Skill": 1, "Block Bonker Skill": 1, "Archaeology Bundle": 1, "Ascension Bundle": 1, "Arch Ability Card": 4, "Arch Ability Infernal Bonus": -0.1509}, "cards": {"dirt1": 4, "dirt2": 4, "dirt3": 4, "dirt4": 3, "com1": 3, "com2": 3, "com3": 4, "com4": 2, "rare1": 3, "rare2": 3, "rare3": 3, "rare4": 2, "epic1": 3, "epic2": 3, "epic3": 4, "epic4": 2, "leg1": 3, "leg2": 3, "leg3": 4, "leg4": 2, "myth1": 3, "myth2": 3, "myth3": 3, "myth4": 2, "div1": 3, "div2": 3, "div3": 3, "div4": 0}}
                    apply_preset(late_game)
                        
            with col_p3:
                if st.button("🗑️ Factory Reset\n(Wipe All Data)", width="stretch", type="secondary"):
                    apply_preset(reset=True)

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

        STAT_TIPS = {
            'Str': "Increase Flat Damage, Damage +x%, Increases Crit Damage",
            'Agi': "Increase Max Sta, Increase Crit Chance %, Increase Speed Mod Chance %",
            'Per': "Increase Frag Gain, Increase Loot Mod Chance %, Increase Armor Pen (flat)",
            'Int': "Increase Exp Gain, Increase Exp Mod Chance %, Increase Armor Pen (%)",
            'Luck': "Increase Crit Chance %, Increase All Mod Chance %, Increase Gold Crosshair Chance %",
            'Div': "Increase Flat Damage, Increase Super Crit Chance %, Increase Crosshair Auto-Tap Chance %",
            'Corr': "Increase Damage (%), Decrease Max Sta (-%), Increase All Mod Multipliers (%)"
        }

        def render_stat(label, stat_key):
            max_val = int(STAT_CAPS[stat_key])
            current_val = int(p.base_stats.get(stat_key, 0))
            safe_val = min(max(current_val, 0), max_val)
            widget_key = f"stat_{stat_key}"
            
            if widget_key not in st.session_state:
                st.session_state[widget_key] = safe_val
                
            with st.container(border=True):
                # Centered Title with Native Browser Tooltip
                tooltip_text = STAT_TIPS.get(stat_key, "")
                st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'><span title='{tooltip_text}' style='cursor: help;'><b>{label}</b> ℹ️</span><br><small>(Max: {max_val})</small></div>", unsafe_allow_html=True)
                
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
            if p.asc1_unlocked:
                render_stat("Divine", 'Div')
            else:
                p.base_stats['Div'] = 0
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
            # --- HIDE MAXED TOGGLE ---
            hide_maxed = st.toggle("👀 Hide Maxed Upgrades", value=False)
            st.divider()

            asc1_locked_rows =[12, 17, 24, 32, 40, 47, 48, 49, 50, 51, 53, 54]
            asc2_locked_rows =[19, 27, 34, 46, 52, 55]
            
            # 1. Pre-filter active upgrades
            active_upgrades = list()
            for upg_id, upg_data in p.UPGRADE_DEF.items():
                if not p.asc1_unlocked and upg_id in asc1_locked_rows: continue
                if not p.asc2_unlocked and upg_id in asc2_locked_rows: continue
                    
                max_lvl = int(cfg.INTERNAL_UPGRADE_CAPS.get(upg_id, 99))
                current_lvl = int(p.upgrade_levels.get(upg_id, 0))
                
                if hide_maxed and current_lvl >= max_lvl:
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
                
                # Hide Hestia Idol from this tab completely (Moved to Arch Idols Tab)
                if group['id'] == 'hestia': continue

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
                                    # Convert decimal to percentage for UI and explicitly store as String for text_input!
                                    st.session_state[inf_key] = str(round(p.arch_ability_infernal_bonus * 100.0, 2))
                                elif not isinstance(st.session_state[inf_key], str):
                                    # Failsafe: coerce any legacy floats stuck in memory into strings
                                    st.session_state[inf_key] = str(st.session_state[inf_key])
                                    
                                def update_inf(k=inf_key):
                                    try:
                                        # Safely parse text to float to avoid +/- confusion
                                        val = float(st.session_state[k])
                                    except ValueError:
                                        val = 0.0
                                    p.arch_ability_infernal_bonus = val / 100.0
                                    p.set_external_level(20, 4) # Trigger W20 math refresh!
                                    
                                st.text_input(
                                    "Inf Bonus", key=inf_key, 
                                    on_change=update_inf, label_visibility="collapsed"
                                )

    # --- TAB 3: BLOCK CARDS ---
    with tab_cards:
        # --- INFERNAL CARD MULTIPLIER UI ---
        with st.container(border=True):
            col_inf_input, col_inf_metric = st.columns(2)
            with col_inf_input:
                if "set_total_inf" not in st.session_state:
                    st.session_state["set_total_inf"] = int(p.total_infernal_cards)
                    
                p.total_infernal_cards = st.number_input(
                    "Total Infernal Cards (Global)", 
                    min_value=0, step=1, key="set_total_inf",
                    help="Sum of all Infernal cards you own across all categories (Archaeology, Fishing, etc). Used for the Infernal Multiplier."
                )
            with col_inf_metric:
                st.metric("🔥 Infernal Arch Card Bonus", f"{p.infernal_multiplier:,.4f}x", help="Mathematically calculated multiplier based on Arch Infernal Cards, Global Infernal Cards, and Hades Idol.")
        
        st.divider()

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
        
        # Build the ordered list: Type then Tier
        ordered_card_ids =[]
        for o_type in block_types:
            for tier_num in range(1, 5):
                ordered_card_ids.append(f"{o_type}{tier_num}")
                
        # Loop over the cards in chunks of 9 columns
        for row_start in range(0, len(ordered_card_ids), 9):
            chunk = ordered_card_ids[row_start:row_start+9]
            cols_cards = st.columns(9)
            
            for col_idx, card_id in enumerate(chunk):
                widget_key = f"card_{card_id}"
                
                current_lvl = int(p.cards.get(card_id, 0))
                if widget_key not in st.session_state:
                    st.session_state[widget_key] = current_lvl
                    
                with cols_cards[col_idx]:
                        with st.container(border=True):
                            # Title
                            st.markdown(f"<div style='text-align: center; margin-bottom: 5px;'><b>{card_id.capitalize()}</b></div>", unsafe_allow_html=True)
                            
                            # --- DETERMINE LOCK STATE ---
                            is_locked = False
                            if card_id.endswith('4') and not p.asc2_unlocked: is_locked = True
                            if card_id.startswith('div') and not p.asc1_unlocked: is_locked = True
                            
                            # Flush invalid states if they toggle Asc2/Asc1 off
                            if is_locked and st.session_state[widget_key] != 0:
                                st.session_state[widget_key] = 0
                            elif not is_locked and not p.asc1_unlocked and st.session_state[widget_key] == 4:
                                st.session_state[widget_key] = 3
                                
                            user_tier = st.session_state[widget_key]
                            p.set_card_level(card_id, user_tier)
                            
                            # --- DYNAMIC CARD COMPOSITING ---
                            if user_tier > 0 and not is_locked:
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
                            
                            max_card_level = 4 if p.asc1_unlocked else 3
                            
                            st.number_input(
                                f"Lvl##{card_id}", min_value=0, max_value=max_card_level,
                                key=widget_key, step=1,
                                on_change=update_card_level, args=(widget_key, card_id),
                                label_visibility="collapsed",
                                disabled=is_locked
                            )

    # --- SUB-TAB: ARCH IDOLS ---
    with tab_idols:
        st.markdown("<div style='text-align: center; margin-bottom: 20px;'><h3>🗿 Arch Idols</h3><p>Manage your Early/Late-Game Asc1 Idols here.</p></div>", unsafe_allow_html=True)
        
        if not p.asc1_unlocked:
            st.warning("🔒 Arch Idols are locked until Ascension 1.")
        else:
            col_hestia, col_hades, _ = st.columns([1, 1, 2])
            
            with col_hestia:
                with st.container(border=True):
                    st.markdown("<div style='text-align: center; margin-bottom: 5px;'><b>Hestia Idol</b></div>", unsafe_allow_html=True)
                    img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "idols", "hestia_idol.png")
                    if os.path.exists(img_path): render_centered_image(img_path, UI_EXT_IMG_STD)
                    st.divider()
                    
                    if "ext_hestia" not in st.session_state: st.session_state["ext_hestia"] = int(p.external_levels.get(4, 0))
                    st.number_input(
                        "Hestia Level", min_value=0, step=1, key="ext_hestia", 
                        on_change=update_external_group, args=("ext_hestia", [4]), label_visibility="collapsed"
                    )

            with col_hades:
                with st.container(border=True):
                    st.markdown("<div style='text-align: center; margin-bottom: 5px;'><b>Hades Idol</b></div>", unsafe_allow_html=True)
                    img_path = os.path.join(ROOT_DIR, "assets", "upgrades", "idols", "hades_idol.png")
                    if os.path.exists(img_path): render_centered_image(img_path, UI_EXT_IMG_STD)
                    else: st.markdown("<div style='text-align: center; color: gray; height: 120px; display: flex; align-items: center; justify-content: center;'>(Missing Asset)</div>", unsafe_allow_html=True)
                    st.divider()
                    
                    if "set_hades" not in st.session_state: st.session_state["set_hades"] = int(p.hades_idol_level)
                    def update_hades(): p.hades_idol_level = st.session_state["set_hades"]
                    st.number_input(
                        "Hades Level", min_value=0, step=1, key="set_hades", 
                        on_change=update_hades, label_visibility="collapsed"
                    )

    # --- TAB 4: CALCULATED STATS ---
    with tab_calc_stats:
        st.subheader("📋 Calculated Player Stats")
        st.write("This is the exact mathematical output derived from your Base Stats, Upgrades, and Cards being fed into the Engine.")
        st.info("💡 **Verification Step:** The best way to ensure the AI gives you perfect results is to verify your inputs! Compare these numbers directly against the stats shown on your in-game Archaeology screen. If they match perfectly, your imported data is correct.")
        
        with st.expander("🛠️ Stat Troubleshooter (Click here if your UI numbers don't match the game!)", expanded=False):
            st.markdown("If a stat in the UI is **higher** than your game, you likely entered an upgrade level too high, allocated too many base stats, or forgot to account for an unequipped pet/skin. Select a mismatched stat below to pull up your **exact current inputs** for that formula:")
            
            troubleshoot_stat = st.selectbox(
                "Select mismatched stat:",["(Select a Stat...)", "Damage", "Armor Pen", "Max Stamina", "Crit Chances & Multipliers", "EXP & Fragment Gain", "Mod Chances & Multipliers", "Abilities (Instacharge / Cooldowns)"],
                label_visibility="collapsed"
            )
            
            if troubleshoot_stat != "(Select a Stat...)":
                # Hardcoded Dependency Maps based on Player.py formulas
                TROUBLESHOOT_MAP = {
                    "Damage": {"stats":["Str", "Corr", "Div"], "upgs":[9, 15, 20, 25, 32, 34, 36, 47, 49, 51, 52], "exts":["Dino Skin", "Hestia Idol"], "infs":["rare2", "div1"]},
                    "Armor Pen": {"stats": ["Per", "Int"], "upgs":[10, 17, 29, 33, 36], "exts": [], "infs":["leg3", "rare3"]},
                    "Max Stamina": {"stats": ["Agi", "Corr"], "upgs":[3, 14, 23, 26, 28, 39, 54], "exts": [], "infs":["epic3"]},
                    "Crit Chances & Multipliers": {"stats":["Luck", "Div"], "upgs":[13, 18, 20, 30, 37, 40, 47, 49, 53], "exts":[], "infs":["com1", "com2", "com3", "epic2"]},
                    "EXP & Fragment Gain": {"stats": ["Int", "Per", "Div"], "upgs":[4, 11, 21, 28, 35, 42, 45, 51], "exts": ["Axolotl Skin", "Geoduck Tribute"], "infs":["dirt2", "dirt3", "leg1"]},
                    "Mod Chances & Multipliers": {"stats":["Luck", "Div", "Corr"], "upgs":[5, 14, 16, 23, 24, 26, 33, 35, 38, 40, 43, 44, 48, 50, 52, 53, 54, 55], "exts": ["Archaeology Bundle"], "infs":["dirt1", "rare1", "epic1", "leg2", "myth2", "myth3", "div3"]},
                    "Abilities (Instacharge / Cooldowns)": {"stats": ["Int", "Div"], "upgs":[18, 22, 29, 31, 32, 39, 50], "exts":["Arch Ability Card", "Avada Keda- Skill", "Block Bonker Skill"], "infs":[]}
                }
                
                data = TROUBLESHOOT_MAP[troubleshoot_stat]
                
                # Dynamically extract all live external upgrade values
                ext_vals = {}
                for group in cfg.EXTERNAL_UI_GROUPS:
                    ext_vals[group['name']] = int(p.external_levels.get(group['rows'][0], 0))
                
                t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                
                with t_col1:
                    st.markdown("##### 📊 Base Stats")
                    for s in data["stats"]:
                        if s == 'Corr' and not p.asc2_unlocked: continue
                        val = int(p.base_stats.get(s, 0))
                        st.markdown(f"**{s}:** `{val}`")
                        
                with t_col2:
                    st.markdown("##### ⬆️ Internal Upgrades")
                    asc2_locked_rows =[19, 27, 34, 46, 52, 55]
                    for u in data["upgs"]:
                        if not p.asc2_unlocked and u in asc2_locked_rows: continue
                        name = p.UPGRADE_DEF.get(u, [f"Upg {u}"])[0]
                        val = int(p.upgrade_levels.get(u, 0))
                        st.markdown(f"**{name}:** `{val}`")
                        
                with t_col3:
                    st.markdown("##### 🌟 External Upgrades")
                    if not data["exts"]:
                        st.markdown("*(None apply)*")
                    else:
                        for e in data["exts"]:
                            val = ext_vals.get(e, 0)
                            st.markdown(f"**{e}:** `{val}`")
                            
                with t_col4:
                    st.markdown("##### 🎴 Infernal Cards")
                    if not data.get("infs"):
                        st.markdown("*(None apply)*")
                    else:
                        for c in data["infs"]:
                            val = p.inf(c)
                            card_label = c.capitalize()
                            # Differentiate between Flat bonuses (decimals=0) and Percentage multipliers
                            if c in ['rare2', 'leg3', 'div3']:
                                st.markdown(f"**{card_label}:** `+{val:,.1f}`")
                            else:
                                st.markdown(f"**{card_label}:** `+{val*100:,.2f}%`")
                            
                # Add conditional warning for common traps
                if troubleshoot_stat == "Damage":
                    st.warning("💡 **Tip:** Did you accidentally input Axolotl levels for Dino Skin levels?")
                elif troubleshoot_stat == "Armor Pen":
                    st.warning("💡 **Tip:** Make sure your Intelligence matches exactly! The percentage-based scaling causes massive variations.")
                elif troubleshoot_stat == "EXP & Fragment Gain":
                    st.warning("💡 **Tip:** Don't forget to check your 'Total Infernal Cards' global input in the External Upgrades tab!")
                    
        st.divider()
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
                st.write(f"**Speed Mod Chance:** {p.speed_mod_chance*100:,.2f}% *(Mod Gain: +{p.speed_mod_gain:,.0f} 2x spd atks)*")
                st.write(f"**Stamina Mod Chance:** {p.stamina_mod_chance*100:,.2f}% *(Mod Gain: +{p.stamina_mod_gain:,.0f} Stamina)*")
                st.divider()
                st.write(f"**Crosshair Auto-Tap Chance:** {p.crosshair_auto_tap*100:,.2f}%")
                st.write(f"**Gold Crosshair Chance:** {p.gold_crosshair_chance*100:,.2f}% *(Mult: {p.gold_crosshair_mult:,.2f}x)*")

        with col_calc_3:
            with st.container(border=True):
                st.markdown("#### ⚡ Abilities")
                st.write(f"**Instacharge Chance:** {p.ability_insta_charge*100:,.2f}%")
                st.divider()
                st.write(f"**Enrage:** {p.enrage_charges:,.0f} charges *(CD: {p.enrage_cooldown:,.0f}s)*")
                st.write(f"*- Dmg Bonus:* +{p.enrage_bonus_dmg*100:,.0f}%")
                st.write(f"*- Enraged Dmg:* {p.enraged_damage:,.0f}")
                st.write(f"*- Crit Bonus:* +{p.enrage_bonus_crit_dmg*100:,.0f}%")
                st.write(f"*- Enraged Crit Dmg:* {p.enraged_crit_dmg_mult:,.2f}x")
                st.divider()
                st.write(f"**Flurry:** {p.flurry_duration:,.0f}s *(CD: {p.flurry_cooldown:,.0f}s)*")
                st.write(f"*- Stamina Gain:* {p.flurry_sta_on_cast:,.0f}")
                st.write(f"*- +100% Atk Speed")
                st.divider()
                st.write(f"**Quake:** {p.quake_attacks:,.0f} atks *(CD: {p.quake_cooldown:,.0f}s)*")
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
            if not p.asc1_unlocked and block_id.startswith('div'): continue
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
        st.markdown("""
        💡 **What is a Breakpoint?**  
        A breakpoint is the exact stat number required to reduce the hits needed to break a block (e.g., dropping from 3 hits down to 2). Because blocks can only take whole hits, any stat points you spend that *don't* push you past the next breakpoint are mathematically wasted!
        
        In the early and mid-game, players use this tool to engineer their stats manually. The goal is to ensure you can kill your target blocks in **1 Regular Hit** (Max Hits), or at least 1 hit if a Critical Strike lands. If you are using the Optimizer, you can use this page to manually verify *why* the AI chose the stats it did!
        """)
        
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
                col_sb1, col_sb2 = st.columns(2)
                with col_sb1:
                    if st.button("🔄 Pull Global Stats", width="stretch", help="Pull your currently saved global stat distribution into the sandbox."):
                        for stat in STAT_CAPS.keys():
                            st.session_state[f"sandbox_stat_{stat}"] = int(p.base_stats.get(stat, 0))
                        st.rerun()
                with col_sb2:
                    def push_sandbox_to_global():
                        # Auto-clamp constraints
                        sandbox_total = sum(int(st.session_state.get(f"sandbox_stat_{s}", 0)) for s in STAT_CAPS.keys())
                        allowed = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
                        if sandbox_total > allowed:
                            st.toast(f"❌ Cannot push: Sandbox uses {sandbox_total} points but budget is {allowed}!", icon="⚠️")
                            return
                        # Safely inject into base stats
                        for s in STAT_CAPS.keys():
                            val = int(st.session_state.get(f"sandbox_stat_{s}", 0))
                            st.session_state[f"stat_{s}"] = val
                            p.base_stats[s] = val
                        st.toast("✅ Sandbox stats pushed to Global UI!", icon="📤")

                    st.button("📤 Push to Global", width="stretch", on_click=push_sandbox_to_global, help="Apply these sandbox stats to your actual character build.")

                st.divider()
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
                if p.asc1_unlocked:
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
                if block_id.startswith('div') and not sandbox_p.asc1_unlocked: continue
                
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
                    "Reg Hit": int(reg_hit),
                    "Crit": int(reg_hit * c_crit),
                    "sCrit": int(reg_hit * c_scrit),
                    "uCrit": int(reg_hit * c_ucrit),
                    "Max Hits": int(max_sta),
                    "Avg Hits": int(avg_sta),
                    "Enr Hit": int(enr_hit),
                    "Enr Crit": int(enr_hit * c_enr_crit),
                    "Enr sCrit": int(enr_hit * c_enr_scrit),
                    "Enr uCrit": int(enr_hit * c_enr_ucrit),
                    "Enr Max Hits": int(max_enr_sta),
                    "Enr Avg Hits": int(avg_enr_sta)
                }
                    
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

                # --- HEADER AND DYNAMIC CRIT TOGGLE ---
                col_bp_title, col_bp_tog = st.columns([1, 1])
                with col_bp_title:
                    st.markdown(f"#### 🎯 Target Breakpoints <span style='font-size: 0.6em; color: gray;'>({len(df_sandbox)} Blocks Displayed)</span>", unsafe_allow_html=True)
                with col_bp_tog:
                    # Use a visually padded container to align the toggle seamlessly next to the title
                    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                    show_crit_details = st.toggle("🔍 Show Detailed Crit Damage Amounts", value=False)
                
                if not show_crit_details:
                    df_sandbox = df_sandbox.drop(columns=["Crit", "sCrit", "uCrit", "Enr Crit", "Enr sCrit", "Enr uCrit"])

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
                    width="stretch",
                    height=700 # Forces the table to be nice and tall to match the left panel
                )
            
    # --- TAB 7: RUN OPTIMIZER ---
    with tab_optimizer:
        st.header("🚀 Monte Carlo Stat Optimizer")
        st.write("Leverage Successive Halving to find the absolute mathematically perfect stat distribution.")

        # --- WORKFLOW GUIDE (SCOUT & REFINE) ---
        with st.container(border=True):
            st.markdown("#### 💡 Best Practice: The 2-Step Optimization")
            st.markdown("""
            **1. The Scout Run:** Leave your stats unlocked. Adjust the **Time Limit** slider below until the Precision Gauge shows at least 🟡 **Moderate**. Run the Optimizer and look at the winning build. Did the AI drop any stats to `0`? Did it push any to their `Max`? \n\n
            **2. The Refined Run:** Open the **Stat Constraints** below and lock those obvious stats to `0` or `Max`. Notice how your Precision Gauge instantly turns 🟢 **Green**! By locking just 1 or 2 stats, the AI can scan the remaining stats with better precision in a fraction of the time. Run it again to find your perfect build.
            """)
            
        # --- PROJECTION DISCLAIMER ---
        with st.expander("ℹ️ How accurate are these projections?", expanded=False):
            disclaimer_text = (
                "**The Good News:** The environment generation in this engine is now **100% identical** to the live game's source code! "
                "The stat distributions this tool provides are mathematically perfect for your current upgrades.\n\n"
                "**The Reality Check #1:** While the combat math is exact, the absolute output numbers (Max Floor, Kills/hr) are built on **Statistical Averages**. "
                "The AI runs hundreds of simulations and optimizes for *consistent, reliable farming*. Because it smooths out extreme RNG, "
                "the engine maintains a slightly conservative slant. Treat these numbers as your highly accurate, reliable baseline!\n\n"
                "**The Reality Check #2:** The engine calculates **100% Theoretical Efficiency**. In the Python simulator, 0.000 seconds pass between killing an ore and hitting the next one. "
                "In the actual live game, minor animation delays and frame drops consume fractions of a second. "
                "Expect your actual real-world Yields to be roughly **~5% to 10% lower** than the mathematical perfection projected here."
            )
        
            # Append the specific warning if Asc2 is checked
            if p.asc2_unlocked:
                disclaimer_text += "\n\n🌌 **Ascension 2 Note:** Because Asc2 unlocks the *Corruption* stat, the AI must search an entire extra dimension of math. Optimizations will naturally take longer to compute than Asc1 runs!"
            
    
            st.warning(disclaimer_text)

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

        # --- DYNAMIC TUTORIAL TIPS ---
        if opt_goal == "Max Floor Push":
            st.info("💡 **Strategy Tip:** Pushing deep floors requires balancing Damage, Armor Pen, Max Stamina and Crits. To force the AI to scan at an extreme precision, try opening the **Stat Constraints** below and locking **Intelligence** to `0` and **Luck** to your max stat cap!")
        elif opt_goal in["Fragment Farming", "Block Card Farming", "Max EXP Yield"]:
            st.info("💡 **Strategy Tip:** If your target spawns on early floors (e.g., Dirt), you don't need Max Stamina or Armor Pen to reach it! Lock **Agility** and **Perception** to `0` to massively increase the precision of the AI's search.\n\n⚠️ **Wait, what if my target is late-game?** If you are farming Tier 4 blocks (which spawn on Floor 81+), you STILL have to survive the gauntlet of tough ores to get there. Do not lock your survival stats to `0`, or the AI will die before reaching your target!")
        st.divider()

        # --- NEW: STAT LOCKING ---
        with st.expander("🔒 Stat Constraints / Locking (Optional)", expanded=False):
            st.markdown("Locking a stat removes an entire dimension from the AI's search grid. **Rule of Thumb:** For every stat you lock, the Precision Gauge below will massively improve, allowing you to run shorter time limits with perfect accuracy!")
            
            def render_lock_stat(label, stat_key, col):
                max_val = int(STAT_CAPS.get(stat_key, 99))
                check_key = f"lock_check_{stat_key}"
                val_key = f"lock_val_{stat_key}"
                
                if check_key not in st.session_state: st.session_state[check_key] = False
                if val_key not in st.session_state: st.session_state[val_key] = int(p.base_stats.get(stat_key, 0))
                
                with col:
                    with st.container(border=True):
                        st.markdown(f"<div style='text-align: center; margin-bottom: 5px; font-size: 0.9em;'><b>{label}</b></div>", unsafe_allow_html=True)
                        
                        img_path_small = os.path.join(ROOT_DIR, "assets", "stats_small", f"{stat_key.lower()}.png")
                        img_path_large = os.path.join(ROOT_DIR, "assets", "stats", f"{stat_key.lower()}.png")
                        if os.path.exists(img_path_small):
                            render_centered_image(img_path_small, 40)
                        elif os.path.exists(img_path_large):
                            render_centered_image(img_path_large, 40)
                            
                        # Controls
                        st.checkbox("Lock Value:", key=check_key)
                        st.number_input(
                            f"Val##{stat_key}", key=val_key, min_value=0, max_value=max_val, step=1,
                            label_visibility="collapsed", disabled=not st.session_state[check_key]
                        )

            lcol1, lcol2, lcol3, lcol4 = st.columns(4)
            render_lock_stat("Strength", 'Str', lcol1)
            render_lock_stat("Agility", 'Agi', lcol2)
            render_lock_stat("Perception", 'Per', lcol3)
            render_lock_stat("Intelligence", 'Int', lcol4)
            render_lock_stat("Luck", 'Luck', lcol1)
            if p.asc1_unlocked:
                render_lock_stat("Divine", 'Div', lcol2)
            if p.asc2_unlocked:
                render_lock_stat("Corruption", 'Corr', lcol3)

        st.divider()

        # --- HARDWARE BENCHMARKING & ETA ---
        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck']
        if p.asc1_unlocked: STATS_TO_OPTIMIZE.append('Div')
        if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
        
        if st.session_state.get("sims_per_sec", 0) == 0:
            # Smart Default: Multiprocessing overhead penalizes short micro-benchmarks.
            # Instead of a blocking spinner, we use an OS baseline that will 
            # invisibly self-correct to 100% accuracy at the end of their very first run.
            if sys.platform == "linux":
                # Humble default for Streamlit Community Cloud. 
                # Free Linux containers only provide 1 vCPU and ~1GB RAM. 
                # Starting at 50 prevents the Auto-Scaler from over-promising on the first run.
                st.session_state.sims_per_sec = 50 
            elif sys.platform == "darwin":
                st.session_state.sims_per_sec = 500
            else:
                st.session_state.sims_per_sec = 800

        # --- LIVE ETA RECALCULATION ---
        DYNAMIC_BUDGET = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
        cap_increase = int(p.u('H45'))
        EFFECTIVE_CAPS = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in STATS_TO_OPTIMIZE}
        
        eta_bounds = {}
        for s in STATS_TO_OPTIMIZE:
            if st.session_state.get(f"lock_check_{s}", False):
                val = int(st.session_state.get(f"lock_val_{s}", 0))
                eta_bounds[s] = (val, val)
            else:
                eta_bounds[s] = (0, EFFECTIVE_CAPS[s])

        st.divider()
        st.markdown("#### ⏱️ Target Compute Time")
        
        TIME_LABELS =["10 Seconds", "30 Seconds", "1 Minute", "2 Minutes", "5 Minutes", "10 Minutes", "30 Minutes (Deep)"]
        TIME_VALUES =[10, 30, 60, 120, 300, 600, 1800]
        
        selected_time_label = st.select_slider(
            "Allocate more time to allow the AI to scan with higher mathematical precision:", 
            options=TIME_LABELS, value="1 Minute", label_visibility="collapsed"
        )
        time_limit_secs = TIME_VALUES[TIME_LABELS.index(selected_time_label)]
        
        # Calculate Engine Execution Plan
        prof_data = get_optimal_step_profile(STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, eta_bounds, st.session_state.sims_per_sec, time_limit_secs)
        step_1 = prof_data['step_1']
        step_2 = prof_data['step_2']
        step_3 = prof_data['step_3']

        # --- DYNAMIC PRECISION GAUGE ---
        if step_1 >= 15:
            g_color, g_bg, g_icon = "#ff4b4b", "rgba(255, 75, 75, 0.1)", "🔴"
            g_title = "Low Precision (Scout Only)"
            g_desc = f"The search grid is too massive. The AI must take huge leaps of <b>{step_1} stat points</b>. This run is only useful for spotting which stats the AI completely ignores. Do not trust the final numbers! Increase time or lock stats."
        elif step_1 >= 5:
            g_color, g_bg, g_icon = "#ffa229", "rgba(255, 162, 41, 0.1)", "🟡"
            g_title = "Moderate Precision"
            g_desc = f"The AI is searching in leaps of <b>{step_1} stat points</b>. It will find a strong general build, but might miss the absolute mathematical peak. Safe to use as a Scout Run."
        else:
            g_color, g_bg, g_icon = "#4CAF50", "rgba(76, 175, 80, 0.1)", "🟢"
            g_title = "High Precision (Recommended)"
            g_desc = f"The search area is extremely tight (leaps of <b>{step_1} stat points</b>). The AI has enough time to pinpoint the mathematically perfect build. Safe to trust!"

        gauge_html = f"""
        <div style='padding: 15px; border: 1px solid {g_color}; border-left: 5px solid {g_color}; background-color: {g_bg}; border-radius: 5px; margin-bottom: 15px;'>
            <div style='font-size: 1.1em; font-weight: bold; margin-bottom: 5px;'>{g_icon} Precision Gauge: {g_title}</div>
            <div style='font-size: 0.9em;'>{g_desc}</div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)

        with st.expander("⚙️ Advanced: Engine Tuning & Hardware Benchmark", expanded=False):
            st.markdown("""
            **🧠 How does the Auto-Scaler work?**
            Testing every stat combination point-by-point would take days. Instead, we "zoom in":
            * **Phase 1 (Coarse):** Casts a wide net across your stat budget in large leaps.
            * **Phase 2 (Fine):** Draws a tight box around the Phase 1 winner and tests smaller leaps.
            * **Phase 3 (Exact):** Pinpoints the mathematical peak by testing every single point in that final box.
            """)
            st.write(f"*(**Execution Plan:** Phase 1 leapes by **{step_1}** -> Phase 2 leaps by **{step_2}** -> Phase 3 leaps by **{step_3}**)*")
            st.divider()
            
            col_spd_1, col_spd_2 = st.columns([3, 1])
            with col_spd_1:
                st.write(f"⚡ **Hardware Speed:** {st.session_state.sims_per_sec:,.0f} sims / second *(Auto-calibrated)*")
            with col_spd_2:
                if st.button("🔄 Reset Calibration", width="stretch"):
                    if "sims_per_sec" in st.session_state: del st.session_state["sims_per_sec"]
                    st.rerun()

    # --- MONTE CARLO EXECUTION LOOP ---
        st.divider()
        
        # Hidden for Production Beta. Change to True if you need to do UI testing later!
        # dev_mode = st.toggle("🛠️ UI Dev Mode (Instantly mock results to design UI without running engine)")
        dev_mode = False
        
        # --- PRE-FLIGHT CHECK ---
        # Calculate total locked points to prevent mathematically impossible runs
        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck']
        if p.asc1_unlocked: STATS_TO_OPTIMIZE.append('Div')
        if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
        DYNAMIC_BUDGET = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
        
        locked_sum = 0
        for s in STATS_TO_OPTIMIZE:
            if st.session_state.get(f"lock_check_{s}", False):
                locked_sum += int(st.session_state.get(f"lock_val_{s}", 0))
                
        if locked_sum > DYNAMIC_BUDGET:
            st.error(f"❌ **Invalid Stat Locks:** You have locked {locked_sum} points, but your total budget is only {DYNAMIC_BUDGET}. Please lower your locked values.")
            btn_disabled = True
        else:
            btn_disabled = False

        st.warning("⚠️ **CRITICAL:** Do not change tabs or click anywhere else on the page while the engine is running! Streamlit will instantly abort the simulation if you interact with the UI.")
        if st.button("🚀 Run Optimizer", use_container_width=True, type="primary", disabled=btn_disabled):
            st.write("---")
            
            # Clean up any stale ROI or Synthesis data from previous runs
            if "roi_stat_results" in st.session_state: del st.session_state["roi_stat_results"]
            if "roi_upg_results" in st.session_state: del st.session_state["roi_upg_results"]
            if "synthesis_result" in st.session_state: del st.session_state["synthesis_result"]
            
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
                    'asc1_unlocked': p.asc1_unlocked, 'asc2_unlocked': p.asc2_unlocked, 
                    'arch_level': p.arch_level,
                    'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                    'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
                    'total_infernal_cards': p.total_infernal_cards
                }

                step_size = step_1
                st.info(f"⏱️ **3-Phase Search Initialized:** Stat Point Leaps (Coarse: **{step_1}**, Fine: **{step_2}**, Exact: **{step_3}**) | Estimated Wall-Clock Time: **{prof_data['time_label']}** (~{prof_data['builds']:,.0f} builds at {st.session_state.sims_per_sec:,.0f} sims/sec)")

                with st.spinner("Engine Running..."):
                    start_time = time.time()
                    STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck']
                    if p.asc1_unlocked: STATS_TO_OPTIMIZE.append('Div')
                    if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                    DYNAMIC_BUDGET = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
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
                    sims_tracker = {}
                    def st_progress_callback(phase_name, r_idx, r_total, task_idx, task_total):
                        # Track simulations per phase AND per round, because task counts reset every halving round
                        tracker_key = f"{phase_name}_R{r_idx}"
                        sims_tracker[tracker_key] = max(sims_tracker.get(tracker_key, 0), task_idx)
                        pct = min(100, max(0, int((task_idx / task_total) * 100)))
                        elapsed_now = time.time() - start_time
                        ui_prog_bar.progress(pct, text=f"⚙️ {phase_name} | Round {r_idx}/{r_total} | {pct}% ({task_idx}/{task_total} sims) | ⏱️ Elapsed: {elapsed_now:.1f}s / {time_limit_secs}s limit")
                    
                    # time_limit_secs is globally defined by the select_slider above
                    
                    import traceback
                    try:
                        # Ensures any points stripped by the Modulo Aligner are placed back into the build
                        def top_up_build(build):
                            if not build: return build
                            b = build.copy()
                            current_sum = sum(b.get(s, 0) for s in STATS_TO_OPTIMIZE)
                            missing = DYNAMIC_BUDGET - current_sum
                            
                            if missing > 0:
                                # SMART TOP-UP: Sort unlocked stats by their current allocation (descending).
                                # By dumping remainder points into the stat the AI already deemed "most valuable", 
                                # we mathematically honor the optimization curve instead of guessing randomly.
                                unlocked_stats =[s for s in STATS_TO_OPTIMIZE if not st.session_state.get(f"lock_check_{s}", False)]
                                unlocked_stats.sort(key=lambda s: b.get(s, 0), reverse=True)
                                
                                for s in unlocked_stats:
                                    if missing <= 0: break
                                    room = EFFECTIVE_CAPS[s] - b.get(s, 0)
                                    if room > 0:
                                        add = min(room, missing)
                                        b[s] += add
                                        missing -= add
                            return b

                        with mp.Pool(CPU_CORES) as pool:
                            # --- PHASE 1 (Coarse) ---
                            bounds_p1 = {}
                            locked_sum_p1 = 0
                            for s in STATS_TO_OPTIMIZE:
                                if st.session_state.get(f"lock_check_{s}", False):
                                    val = int(st.session_state.get(f"lock_val_{s}", 0))
                                    bounds_p1[s] = (val, val)
                                    locked_sum_p1 += val
                                else:
                                    bounds_p1[s] = (0, EFFECTIVE_CAPS[s])
                                    
                            if locked_sum_p1 > DYNAMIC_BUDGET:
                                raise ValueError(f"You locked {locked_sum_p1} points, but your global budget is only {DYNAMIC_BUDGET}.")
                                
                            # Modulo Aligner: Ensures the grid perfectly divides the remaining budget
                            rem_p1 = (DYNAMIC_BUDGET - locked_sum_p1) % step_size
                            p1_budget = DYNAMIC_BUDGET - rem_p1
                                    
                            best_p1, summary_p1 = run_optimization_phase(
                                "Phase 1 (Coarse)", target_metric, STATS_TO_OPTIMIZE, 
                                p1_budget, step_size, ITER_P1, pool, FIXED_STATS, bounds_p1,
                                progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                                base_state_dict=base_state_dict
                            )
                            best_p1 = top_up_build(best_p1)
                            
                            # --- PHASE 2 (Fine) ---
                            best_p2, summary_p2 = None, None
                            if best_p1 and (time.time() - start_time) < time_limit_secs:
                                bounds_p2 = {}
                                step_2 = max(2, step_size // 3)
                                locked_sum_p2 = 0
                                for s in STATS_TO_OPTIMIZE:
                                    if st.session_state.get(f"lock_check_{s}", False):
                                        bounds_p2[s] = bounds_p1[s]
                                        locked_sum_p2 += bounds_p1[s][0]
                                    else:
                                        bounds_p2[s] = (max(0, best_p1[s] - step_size), min(EFFECTIVE_CAPS[s], best_p1[s] + step_size))
                                        
                                rem_p2 = (DYNAMIC_BUDGET - locked_sum_p2) % step_2
                                p2_budget = DYNAMIC_BUDGET - rem_p2
                                        
                                best_p2, summary_p2 = run_optimization_phase(
                                    "Phase 2 (Fine)", target_metric, STATS_TO_OPTIMIZE, 
                                    p2_budget, step_2, ITER_P2, pool, FIXED_STATS, bounds_p2,
                                    progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                                    base_state_dict=base_state_dict
                                )
                                best_p2 = top_up_build(best_p2)
                                
                            if best_p2 and (time.time() - start_time) < time_limit_secs:
                                bounds_p3 = {}
                                # Read the dynamically calculated Phase 3 constraints from the Auto-Scaler!
                                p3_radius = prof_data.get('p3_radius', min(2, step_2)) 
                                step_3 = prof_data.get('step_3', 1)
                                
                                for s in STATS_TO_OPTIMIZE:
                                    if st.session_state.get(f"lock_check_{s}", False):
                                        bounds_p3[s] = bounds_p1[s]
                                    else:
                                        bounds_p3[s] = (max(0, best_p2[s] - p3_radius), min(EFFECTIVE_CAPS[s], best_p2[s] + p3_radius))
                                        
                                best_p3, final_summary = run_optimization_phase(
                                    f"Phase 3 (Radius ±{p3_radius})", target_metric, STATS_TO_OPTIMIZE, 
                                    DYNAMIC_BUDGET, step_3, ITER_P3, pool, FIXED_STATS, bounds_p3,
                                    progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                                    base_state_dict=base_state_dict
                                )
                                best_p3 = top_up_build(best_p3)
                    except Exception as crash_err:
                        st.error(f"🚨 **CRITICAL ENGINE CRASH:** The background worker was unexpectedly killed. This is almost always caused by the Streamlit Cloud server temporarily running out of memory.")
                        st.error(f"**Technical Details:** `{type(crash_err).__name__}: {str(crash_err)}`")
                        with st.expander("View Full Crash Traceback", expanded=False):
                            st.code(traceback.format_exc(), language="python")
                        print(f"\n[CRASH LOG] {traceback.format_exc()}\n")
                        best_p3, final_summary = None, None
                    except BaseException as interrupt_err:
                        print(f"\n[INTERRUPT LOG] Script halted during Execution Loop! Reason: {type(interrupt_err).__name__}")
                        print("-> This is usually caused by the user clicking a button, changing a tab, or a browser disconnection.\n")
                        raise # We MUST re-raise this so Streamlit can correctly stop the thread!
                    
                    best_final = best_p3 or best_p2 or best_p1
                    final_summary_out = final_summary or summary_p2 or summary_p1
                    ui_prog_bar.empty()
                    elapsed = time.time() - start_time
                    
                    # --- AUTO-CALIBRATE TRUE HARDWARE SPEED ---
                    # Now that a deep run is complete, calculate the true hardware speed.
                    # This guarantees the ETA for all future runs is flawlessly accurate!
                    total_sims_executed = sum(sims_tracker.values())
                    if elapsed > 0 and total_sims_executed > 0:
                        st.session_state.sims_per_sec = max(1, int(total_sims_executed / elapsed))
                    
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
            # SAVE TO SESSION STATE FOR PERSISTENCE
            # ==========================================
            if best_final and final_summary_out:
                # --- NEW: APPEND TO RUN HISTORY ---
                if "run_history" not in st.session_state:
                    st.session_state.run_history = list()
                
                history_entry = {
                    "Include": True,
                    "Target": target_metric,
                    "Metric Score": round(final_summary_out.get(target_metric, 0), 2),
                    "Avg Floor": round(final_summary_out.get("avg_floor", 0), 2),
                    "Max Floor": int(final_summary_out.get("abs_max_floor", 0))
                }
                history_entry.update(best_final) # Add the stats to the dictionary
                st.session_state.run_history.append(history_entry)

                st.session_state.opt_results = {
                    "best_final": best_final,
                    "final_summary_out": final_summary_out,
                    "elapsed": elapsed,
                    "time_limit_secs": time_limit_secs,
                    "run_target_metric": target_metric,
                    "worst_val": worst_val,
                    "avg_val": avg_val,
                    "runner_up_val": runner_up_val,
                    "chart_hill_scores": chart_hill_scores,
                    "chart_hill_labels": chart_hill_labels,
                    "chart_hist": chart_hist,
                    "chart_loot": chart_loot,
                    "show_loot": (target_metric != "highest_floor" or dev_mode),
                    "show_wall": (target_metric == "highest_floor" or dev_mode)
                }
            else:
                st.error("Optimization failed or aborted before a single build could be tested.")

        # ==========================================
        # UI RESULTS TELEMETRY (PERSISTENT RENDER)
        # ==========================================
        # Because this is OUTSIDE the st.button() block, it will survive tab changes!
        
        def cb_apply_stats(target, stats_dict, msg, icon):
            """Streamlit callback to securely inject state before UI rendering."""
            for k, v in stats_dict.items():
                if target == "global":
                    st.session_state[f"stat_{k}"] = int(v)
                    st.session_state.player.base_stats[k] = int(v)
                elif target == "sandbox":
                    st.session_state[f"sandbox_stat_{k}"] = int(v)
            st.toast(msg, icon=icon)

        def cb_delete_hist(index):
            """Streamlit callback to securely delete history rows."""
            if "synth_history" in st.session_state and index < len(st.session_state.synth_history):
                st.session_state.synth_history.pop(index)
                st.toast("🗑️ Meta-Build permanently deleted!", icon="🧹")

        # Provide a fallback message if they navigate to Synth before running the optimizer
        with tab_synth:
            if "run_history" not in st.session_state or len(st.session_state.run_history) == 0:
                st.info("No simulation history available. Head over to the Optimizer tab and run a Monte Carlo simulation first!")

        if "opt_results" in st.session_state:
            res = st.session_state.opt_results
            
            best_final = res["best_final"]
            final_summary_out = res["final_summary_out"]
            elapsed = res["elapsed"]
            time_limit_secs = res["time_limit_secs"]
            run_target_metric = res["run_target_metric"]
            worst_val = res["worst_val"]
            avg_val = res["avg_val"]
            runner_up_val = res["runner_up_val"]
            chart_hill_scores = res["chart_hill_scores"]
            chart_hill_labels = res["chart_hill_labels"]
            chart_hist = res["chart_hist"]
            chart_loot = res["chart_loot"]
            show_loot = res["show_loot"]
            show_wall = res["show_wall"]

            if elapsed >= time_limit_secs:
                st.warning(f"⚠️ **Time Limit Reached!** Optimization safely aborted early at {elapsed:.1f} seconds. Showing the best build found so far.")
            else:
                st.success(f"✅ Simulation Complete in {elapsed:.1f} seconds!")
                
            # --- POST-RUN RESULTS HIERARCHY ---
            tab_res_build, tab_res_data, tab_res_roi = st.tabs(["🏆 The Build", "📊 Simulation Data", "🔮 Upgrade Guide (ROI)"])
            
            with tab_res_build:
                st.markdown("### 🏆 Optimal Stat Build")
                
                # --- ELI5 DYNAMIC SUMMARY ---
                if run_target_metric == "highest_floor":
                    eli5_target = f"highest mathematical probability to reach **Floor {final_summary_out.get('abs_max_floor', 0):,.0f}**"
                elif "frag" in run_target_metric:
                    eli5_target = "absolute highest **Fragment Farming** yields"
                elif "block" in run_target_metric:
                    eli5_target = "absolute highest **Block Card Farming** yields"
                else:
                    eli5_target = "absolute highest **EXP/min** yields"
                    
                st.info(f"🔥 **Simulation Complete!** The AI determined that shifting your stats to the distribution below gives you the {eli5_target}.")
                st.write("<small>*(Green/Red numbers show changes from your current UI allocation)*</small>", unsafe_allow_html=True)
                
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
                
                col_apply1, col_apply2 = st.columns(2)
                with col_apply1:
                    st.button("✨ Apply Build Globally", width="stretch", on_click=cb_apply_stats, args=("global", best_final, "✅ Optimal stats applied globally!", "🎉"))
                with col_apply2:
                    st.button("🧪 Send to Sandbox", width="stretch", on_click=cb_apply_stats, args=("sandbox", best_final, "✅ Optimal stats piped to Tab 6 (Hit Calculator)!", "🧪"))

            # ==========================================
            # ADVANCED ANALYTICS DASHBOARD (TABS)
            # ==========================================
            with tab_res_data:
                st.markdown("### 📊 Advanced Analytics Dashboard")
                tab_list =["📈 Performance"]
                
                if run_target_metric != "highest_floor" or dev_mode: tab_list.append("🃏 Card Drops")
                if show_loot: tab_list.append("🎒 Loot Breakdown")
                if show_wall: tab_list.append("🧱 Progression Wall")
                
                ui_tabs = st.tabs(tab_list)
                tab_idx = 0
                
                # --- TAB 1: PERFORMANCE & CONFIDENCE ---
                with ui_tabs[tab_idx]:
                    tab_idx += 1
                    perf_col1, perf_col2 = st.columns([1, 1.5])
                    
                    with perf_col1:
                        if run_target_metric == "highest_floor":
                            st.markdown("#### 🏆 Push Potential")
                            
                            # Use the specific telemetry calculated in parallel_worker.py
                            abs_max = final_summary_out.get("abs_max_floor", final_summary_out.get(run_target_metric, 0))
                            abs_chance = final_summary_out.get("abs_max_chance", 0) * 100
                            avg_flr = final_summary_out.get("avg_floor", final_summary_out.get(run_target_metric, 0))
                            
                            st.metric("Theoretical Peak Floor", f"Floor {abs_max:,.0f}")
                            st.metric("Peak Probability", f"{abs_chance:.1f}%")
                            st.metric("Average Consistency Floor", f"Floor {avg_flr:,.1f}")
                        else:
                            val = final_summary_out[run_target_metric]
                            rate_1k = (val / 60.0) * 1000.0
                            metric_str = "Fragments" if "frag" in run_target_metric else "Kills" if "block" in run_target_metric else "EXP"
                            
                            # Elevate the prominence of the Arch Seconds metric
                            st.markdown(f"#### 💰 Banked Yields<br><span style='font-size: 0.9em; color: gray;'>Target {metric_str} per <b>1k Arch Seconds</b></span>", unsafe_allow_html=True)
                            st.metric("Yield", f"{rate_1k:,.1f}", label_visibility="collapsed")
                            
                            st.divider()
                            
                            # Demote real-time yield
                            st.markdown(f"#### ⏱️ Real-Time Yield<br><span style='font-size: 0.9em; color: gray;'>{metric_str} / minute</span>", unsafe_allow_html=True)
                            st.metric("Real-Time", f"{val:,.2f}", label_visibility="collapsed")
                            
                            # --- ⬆️ LEVEL UP CALCULATOR ---
                            if run_target_metric == "xp_per_min":
                                st.divider()
                                st.markdown(f"#### 🆙 Level Up Calculator<br><span style='font-size: 0.9em; color: gray;'>Based on {val:,.2f} EXP/min</span>", unsafe_allow_html=True)
                                
                                col_xp_c, col_xp_t = st.columns(2)
                                with col_xp_c:
                                    cur_xp = st.number_input("Current EXP", min_value=0.0, step=1000.0, format="%.0f", key="perf_cur_xp")
                                with col_xp_t:
                                    tar_xp = st.number_input("Target EXP", min_value=0.0, step=1000.0, format="%.0f", key="perf_tar_xp")
                                    
                                if cur_xp > 0 or tar_xp > 0:
                                    if tar_xp > cur_xp and val > 0:
                                        mins_req = (tar_xp - cur_xp) / val
                                        st.success(f"**Required:** ~{(mins_req * 60.0) / 1000.0:,.1f}k Arch Seconds ({mins_req:,.1f} mins real-time)")
                                    elif tar_xp <= cur_xp:
                                        st.warning("Target EXP must be greater than Current EXP.")
    
                            st.divider()
                            
                            avg_flr = final_summary_out.get("avg_floor", 0)
                            st.markdown(f"#### 🧱 Average Death<br><span style='font-size: 0.9em; color: gray;'>Floor reached per run</span>", unsafe_allow_html=True)
                            st.metric("Avg Floor", f"Floor {avg_flr:,.1f}", label_visibility="collapsed")
    
                    with perf_col2:
                        # UI Transformation: Scale values to 1k Arch Secs for relevant targets
                        is_floor_target = (run_target_metric == "highest_floor")
                        def scale_score(v): return v if is_floor_target else (v / 60.0) * 1000.0
                        
                        unit_label = "Floor Reached" if is_floor_target else "Yield per 1k Arch Secs"
    
                        # Streamlit Markdown header completely fixes the Plotly overlap bug
                        st.markdown(
                            f"#### AI Convergence (Hill Climb)<br><span style='font-size: 0.8em; color: gray;'>Y-Axis: {unit_label}</span> "
                            "<span title='This chart shows how the AI narrowed down the best build across the 3 optimization phases. "
                            "An upward curve means the engine successfully found significantly better builds as it zoomed in. "
                            "A flat line means Phase 1 already hit the near-perfect build.' "
                            "style='cursor: help; font-size: 0.8em;'>ℹ️</span>", 
                            unsafe_allow_html=True
                        )
                        df_hill = pd.DataFrame({"Phase": chart_hill_labels, "Score":[scale_score(s) for s in chart_hill_scores]})
                        fig_hill = px.line(df_hill, x="Phase", y="Score", markers=True)
                        fig_hill.update_traces(line_color='#4CAF50', marker=dict(size=10))
                        fig_hill.update_layout(margin=dict(l=10, r=20, t=10, b=20), height=200)
                        st.plotly_chart(fig_hill, width="stretch")
                        
                        # Streamlit Markdown header
                        st.markdown(
                            f"#### Engine Confidence Analysis<br><span style='font-size: 0.8em; color: gray;'>X-Axis: {unit_label}</span> "
                            "<span title='Compares the Optimal build against the Worst, Average, and Runner-Up builds tested. "
                            "A large gap between Optimal and Average proves your stats highly impact this target. A small gap between Runner-Up and Optimal "
                            "shows the AI fine-tuned the absolute perfect micro-adjustments.' "
                            "style='cursor: help; font-size: 0.8em;'>ℹ️</span>", 
                            unsafe_allow_html=True
                        )
                        df_conf = pd.DataFrame({
                            "Build Category":["Worst Tested", "Average", "Runner-Up", "🏆 Optimal"],
                            "Performance":[scale_score(worst_val), scale_score(avg_val), scale_score(runner_up_val), scale_score(final_summary_out[run_target_metric])]
                        })
                        fig_conf = px.bar(
                            df_conf, x="Performance", y="Build Category", orientation='h', text_auto='.3s', color="Build Category",
                            color_discrete_map={"Worst Tested": "#ff4b4b", "Average": "#ffa229", "Runner-Up": "#6495ED", "🏆 Optimal": "#4CAF50"}
                        )
                        fig_conf.update_layout(showlegend=False, margin=dict(l=10, r=20, t=10, b=20), height=200)
                        st.plotly_chart(fig_conf, width="stretch")
    
                # --- NEW TAB: CARD DROPS ---
                if run_target_metric != "highest_floor" or dev_mode:
                    with ui_tabs[tab_idx]:
                        tab_idx += 1
                        
                        @st.fragment
                        def render_card_drops():
                            st.markdown("#### 🎴 Block Card Drop Estimates")
                            
                            # Extract the exact average kill rates for EVERY block from the telemetry
                            avg_metrics = final_summary_out.get("avg_metrics", {})
                            available_blocks =[k.replace("block_", "").replace("_per_min", "") for k in avg_metrics.keys() if k.startswith("block_")]
                            
                            if not available_blocks:
                                st.info("No block kill data available for this run.")
                            else:
                                # Sort blocks alphabetically
                                available_blocks.sort()
                                
                                # Default to the target block if it was a Card Farming run
                                is_block_run = "block_" in run_target_metric
                                target_block_id = run_target_metric.replace("block_", "").replace("_per_min", "") if is_block_run else None
                                default_idx = available_blocks.index(target_block_id) if target_block_id in available_blocks else 0
                                
                                col_c1, col_c2 = st.columns([1, 2])
                                with col_c1:
                                    # Let the user pick exactly which block they want to inspect
                                    selected_block = st.selectbox(
                                        "Select Block to view Drop Projections:", 
                                        options=available_blocks, 
                                        index=default_idx, 
                                        format_func=lambda x: x.capitalize()
                                    )
                                
                                # Fetch the true kill rate for this specific block
                                val = avg_metrics.get(f"block_{selected_block}_per_min", 0)
                                
                                st.markdown(f"<span style='font-size: 0.9em; color: gray;'>Based on {val:,.2f} <b>{selected_block.capitalize()}</b> kills/min</span>", unsafe_allow_html=True)
                                st.divider()
                                
                                odds = {"Base Card": 1500, "Poly Fragments": 7500, "Infernal Fragments": 200000}
                                cblock_path = os.path.join(ROOT_DIR, "assets", "cards", "cores", f"{selected_block}.png")
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
                                                kills_99 = 4.605 * base_odds
                                                
                                                def format_time(req_kills):
                                                    rt_mins = req_kills / val
                                                    rt_str = f"{rt_mins:.1f}m" if rt_mins < 60 else f"{rt_mins/60.0:.1f}h"
                                                    arch_secs = req_kills / (val / 60.0)
                                                    arch_1k = arch_secs / 1000.0
                                                    return rt_str, arch_1k
        
                                                rt_50, bk_50 = format_time(kills_50)
                                                rt_90, bk_90 = format_time(kills_90)
                                                rt_99, bk_99 = format_time(kills_99)
                                                
                                                st.markdown(f"<small><b>50% Chance (Lucky):</b><br>~{rt_50} | ~{bk_50:.1f}k Arch Seconds</small>", unsafe_allow_html=True)
                                                st.markdown(f"<small><b>90% Chance (Safe):</b><br>~{rt_90} | ~{bk_90:.1f}k Arch Seconds</small>", unsafe_allow_html=True)
                                                st.markdown(f"<small><b>99% Chance (Guaranteed):</b><br>~{rt_99} | ~{bk_99:.1f}k Arch Seconds</small>", unsafe_allow_html=True)
                                            else:
                                                st.markdown("<div style='text-align: center; color: gray;'><small>N/A (0 kills)</small></div>", unsafe_allow_html=True)
    
                        render_card_drops()
    
                # --- TAB 2: COLLATERAL LOOT (BAR CHART) ---
                if show_loot:
                    with ui_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("#### Collateral Loot Distribution")
                        st.write("On average, every **1k Arch Seconds** of simulated mining yields the following collateral fragments alongside your target:")
                        
                        # Transform values from per-minute to per 1k Arch Seconds
                        scaled_loot = {k: (v / 60.0) * 1000.0 for k, v in chart_loot.items()}
                        
                        total_loot = sum(scaled_loot.values()) if scaled_loot else 1
                        df_loot = pd.DataFrame(list(scaled_loot.items()), columns=['Loot Tier', 'Amount'])
                        df_loot['Label'] = df_loot['Amount'].apply(lambda x: f"{x:,.1f}  ({(x/total_loot)*100:.1f}%)")
                        
                        fig_loot = px.bar(
                            df_loot, x='Loot Tier', y='Amount', text='Label', color='Loot Tier',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_loot.update_traces(textposition='outside')
                        fig_loot.update_layout(showlegend=False, margin=dict(t=20, b=20), height=400)
                        st.plotly_chart(fig_loot, width="stretch")
    
                # --- TAB 3: PROGRESSION WALL (HISTOGRAM) ---
                if show_wall:
                    with ui_tabs[tab_idx]:
                        tab_idx += 1
                        st.markdown("#### Death Distribution (Progression Wall)")
                        st.write("Out of the simulations run on the optimal build, this is exactly where your character died. High spikes indicate a hard progression wall (usually enemy armor).")
                        
                        df_hist = pd.DataFrame(list(chart_hist.items()), columns=['Floor', 'Deaths'])
                        # Sort the dataframe by Floor numerically so the x-axis reads chronologically
                        df_hist['Floor'] = pd.to_numeric(df_hist['Floor'])
                        df_hist = df_hist.sort_values(by='Floor')
                        
                        fig_hist = px.bar(df_hist, x='Floor', y='Deaths', text='Deaths')
                        fig_hist.update_traces(marker_color='#ff4b4b', textposition='outside')
                        fig_hist.update_layout(margin=dict(t=20, b=20), height=400, xaxis_type='category')
                        st.plotly_chart(fig_hist, width="stretch")
    
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
                            st.plotly_chart(fig_stam, width="stretch")

            # ==========================================
            # RUN HISTORY & SYNTHESIS (TAB ROUTING)
            # ==========================================
            with tab_synth:
                st.markdown("### 🧬 Build Synthesis & Tie-Breakers")
                st.markdown("Because blocks only take whole hits, multiple different stat builds can tie for 1st place (a **Stat Plateau**). Use this tool to merge your best historical runs and calculate the absolute mathematical peak.")
                
                with st.expander("🤓 Deep Dive: The Stat Plateau & RNG Tie-Breakers"):
                    st.markdown("""
                    * **The Math:** If 50 Strength kills a block in exactly 3 hits, having 54 Strength *also* kills it in 3 hits. This creates a "Stat Plateau" where wildly different builds are mathematically identical.
                    * **The Tie-Breaker (RNG):** To break the tie, the AI forces your selected builds to race 500 times. Whichever tied build happens to get slightly luckier with Critical Hits across a massive sample size wins the gold medal!
                    * **The Synthesis:** The engine calculates the statistical center of your checked builds, generates nearby hybrid combinations, and runs the exhaustive 500-iteration tournament to find the true Meta-Build.
                    
                    **The Takeaway:** If your stats bounce around slightly between 1-minute scout runs, congratulations—you've reached the absolute peak!
                    """)
                st.divider()

                if "run_history" in st.session_state and st.session_state.run_history:
                    # State migration failsafe: Normalize stale runs from older app versions
                    for r in st.session_state.run_history:
                        if "Target" not in r:
                            r["Target"] = "unknown"
                            
                    unique_targets = list(set(r.get("Target") for r in st.session_state.run_history))
                    
                    col_filt1, col_filt2 = st.columns([2, 1])
                    with col_filt1:
                        view_targets = st.multiselect(
                            "🔍 Filter visible runs by optimization target:", 
                            options=unique_targets, 
                            default=[t for t in unique_targets if t == run_target_metric] or unique_targets
                        )
                    with col_filt2:
                        # Add a top margin to perfectly align the button with the multiselect input box
                        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                        if st.button("☑️ Check / Uncheck All Visible", width="stretch", help="Instantly toggle the 'Include' checkboxes for all runs currently shown in the table below."):
                            for r in st.session_state.run_history:
                                if r.get("Target") in view_targets:
                                    r["Include"] = not r.get("Include", True)
                            # Flush data editor memory so it redraws with the new backend boolean values
                            for k in list(st.session_state.keys()):
                                if k.startswith("history_editor_"):
                                    del st.session_state[k]
                            st.rerun()

                    # Inject global index to securely map frontend edits back to the session state array
                    for i, r in enumerate(st.session_state.run_history):
                        r["_global_idx"] = i

                    visible_history =[r for r in st.session_state.run_history if r.get("Target") in view_targets]
                    
                    if not visible_history:
                        st.info("No runs match the selected filters. Run the optimizer to build history.")
                    else:
                        st.markdown("#### 🏆 Run Tie-Breaker Tournament")
                        st.markdown("Once you have checked the **Include** box for your top 2-5 runs in the history table below, click the Synthesize button to merge them into the ultimate Meta-Build.")
                        
                        # Full-width placeholder to pull the progress bar out of the squished columns!
                        synth_progress_ui = st.empty()
                        
                        col_synth1, col_synth2 = st.columns(2)
                        with col_synth1:
                            if st.button("🧬 Synthesize Ultimate Meta-Build", width="stretch"):
                                # WYSIWYG Guard: Only synthesize runs that are currently visible in the UI filter!
                                valid_runs =[r for r in visible_history if r.get("Include", False)]
                                
                                if len(valid_runs) == 0:
                                    st.error("⚠️ You must have at least 1 visible run checked to synthesize!")
                                elif len(valid_runs) > 10:
                                    st.error("⚠️ **Safety Limit Reached:** Synthesizing creates dozens of mathematical permutations for every input build. Please select 10 or fewer builds to prevent server memory overloads!")
                                else:
                                    with st.spinner("Calculating center, generating permutations, and running deep verification..."):
                                        stat_keys =[k for k in valid_runs[0].keys() if k not in["Include", "Target", "Metric Score", "Avg Floor", "Max Floor", "_global_idx"]]
                                        
                                        synth_state_dict = {
                                            'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                                            'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                                            'asc1_unlocked': p.asc1_unlocked, 'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                                            'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                                            'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
                                            'total_infernal_cards': p.total_infernal_cards
                                        }
                                        
                                        if sys.platform == "linux": CPU_CORES = min(2, mp.cpu_count()) 
                                        else: CPU_CORES = max(1, mp.cpu_count() - 1)
                                        
                                        cap_increase = int(p.u('H45'))
                                        EFFECTIVE_CAPS = {s: cfg.BASE_STAT_CAPS[s] + cap_increase for s in stat_keys}

                                        # ==========================================================
                                        # UNIFIED MULTI-SEED TOURNAMENT SYNTHESIS
                                        # ==========================================================
                                        candidates = []
                                        original_b_ids =[]
                                        
                                        # 1. Add original runs
                                        for r in valid_runs:
                                            dist = {s: r[s] for s in stat_keys}
                                            b_id = tuple(dist.items())
                                            if b_id not in original_b_ids: original_b_ids.append(b_id)
                                            if dist not in candidates: candidates.append(dist)
                                            
                                        # 2. Add the Average Build
                                        avg_dist = {}
                                        for s in stat_keys: avg_dist[s] = int(round(sum(r[s] for r in valid_runs) / len(valid_runs)))
                                        diff = sum(valid_runs[-1][s] for s in stat_keys) - sum(avg_dist.values())
                                        if diff != 0: avg_dist[max(stat_keys, key=lambda k: avg_dist[k])] += diff
                                        
                                        avg_b_id = tuple(avg_dist.items())
                                        if avg_dist not in candidates: candidates.append(avg_dist)
                                        
                                        # 3. Smart Mutation: +/- 1 around history runs, +/- 1 & 2 around Average Center
                                        base_dists = candidates.copy()
                                        for base_dist in base_dists:
                                            radii = [1, 2] if tuple(base_dist.items()) == avg_b_id else [1]
                                            for radius in radii:
                                                for s_from in stat_keys:
                                                    if base_dist[s_from] >= radius and not st.session_state.get(f"lock_check_{s_from}", False):
                                                        for s_to in stat_keys:
                                                            if s_from != s_to and base_dist[s_to] <= EFFECTIVE_CAPS[s_to] - radius and not st.session_state.get(f"lock_check_{s_to}", False):
                                                                neighbor = base_dist.copy()
                                                                neighbor[s_from] -= radius
                                                                neighbor[s_to] += radius
                                                                if neighbor not in candidates:
                                                                    candidates.append(neighbor)
                                                                    
                                        # --- ETA & PROGRESS BAR CALCULATION ---
                                        total_r1_sims = len(candidates) * 50
                                        est_r2_count = min(5, len(candidates)) + len(original_b_ids)
                                        total_r2_sims = est_r2_count * 450
                                        total_sims = total_r1_sims + total_r2_sims
                                        
                                        # Fallback to 1000 if user skipped manual benchmark
                                        spd = st.session_state.get('sims_per_sec', 1000)
                                        sims_spd = spd if spd > 0 else 1000
                                        eta_secs = total_sims / sims_spd
                                        
                                        # Teleport the progress UI out of the column and into the full-width placeholder above!
                                        with synth_progress_ui.container():
                                            st.markdown("""
                                            <style>
                                                /* Fade out the stale legacy UI below the buttons to prevent duplicate confusion */
                                                div[data-testid="stVerticalBlock"] > div:has(h4:contains("Synthesis Performance Proof")),
                                                div[data-testid="stVerticalBlock"] > div:has(h4:contains("Synthesized Stat Allocation")),
                                                div[data-testid="stVerticalBlock"] > div:has(h3:contains("Meta-Build History Log")) {
                                                    opacity: 0.1 !important;
                                                    pointer-events: none !important;
                                                }
                                            </style>
                                            <h3 style='text-align: center; color: #ffa229; border: 2px solid #ffa229; padding: 10px; border-radius: 10px; background-color: rgba(255, 162, 41, 0.1); margin-bottom: 20px;'>⚙️ Meta-Build Synthesis in Progress...</h3>
                                            """, unsafe_allow_html=True)
                                            synth_prog = st.progress(0, text=f"🧬 Prepping {len(candidates)} build permutations... (~{total_sims:,} sims | ETA: {eta_secs:.1f}s)")
                                            
                                        synth_start_time = time.time()
                                        
                                        # TOURNAMENT ROUND 1: 50 runs each
                                        r1_args =[ {'stats': b, 'fixed_stats': {}, 'state_dict': synth_state_dict, '_b_id': tuple(b.items())} for b in candidates for _ in range(50) ]
                                        res1 = [ ]
                                        
                                        with mp.Pool(CPU_CORES) as pool:
                                            # Using imap to stream results back and update the progress bar cleanly!
                                            for i, r in enumerate(pool.imap(worker_simulate, r1_args, chunksize=max(1, len(r1_args)//100))):
                                                res1.append(r)
                                                if i % 50 == 0:
                                                    pct = min(100, int((i / total_sims) * 100))
                                                    synth_prog.progress(pct, text=f"⚔️ Round 1/2: Testing {len(candidates)} builds ({i}/{len(r1_args)} sims) | ETA: {eta_secs:.1f}s")
                                            
                                        build_res = {}
                                        for args, r in zip(r1_args, res1):
                                            b_id = args['_b_id']
                                            if b_id not in build_res: build_res[b_id] = {'sum_t': 0.0, 'sum_f': 0, 'floors': [ ]}
                                            
                                            t_val = float(r.get(run_target_metric, 0.0))
                                            f_val = r.get("highest_floor", 0)
                                            
                                            build_res[b_id]['sum_t'] += t_val
                                            build_res[b_id]['sum_f'] += f_val
                                            build_res[b_id]['floors'].append(f_val)
                                            
                                        # SORTING LOGIC FOR ROUND 1
                                        def get_ceiling_score(floors, count=3):
                                            sorted_f = sorted(floors)
                                            return sum(sorted_f[-count:]) / float(count) if floors else 0
                                            
                                        if run_target_metric == "highest_floor":
                                            top5_ids = sorted(build_res.keys(), key=lambda k: get_ceiling_score(build_res[k]['floors'], 3), reverse=True)[:5]
                                        else:
                                            top5_ids = sorted(build_res.keys(), key=lambda k: build_res[k]['sum_t'], reverse=True)[:5]
                                        
                                        # FORCE ORIGINAL RUNS INTO ROUND 2:
                                        # We evaluate all original history runs to 500 simulations to strip away their RNG 
                                        # noise so we can do a fair Apples-to-Apples comparison on the final chart!
                                        r2_ids = list(set(top5_ids + original_b_ids))
                                        
                                        # TOURNAMENT ROUND 2: 450 runs on the finalists & original runs
                                        r2_args =[ {'stats': dict(b_id), 'fixed_stats': {}, 'state_dict': synth_state_dict, '_b_id': b_id} for b_id in r2_ids for _ in range(450) ]
                                        res2 = [ ]
                                        
                                        with mp.Pool(CPU_CORES) as pool:
                                            for i, r in enumerate(pool.imap(worker_simulate, r2_args, chunksize=max(1, len(r2_args)//100))):
                                                res2.append(r)
                                                if i % 50 == 0:
                                                    current_sims = len(r1_args) + i
                                                    pct = min(100, int((current_sims / total_sims) * 100))
                                                    synth_prog.progress(pct, text=f"⚔️ Round 2/2: Deep verifying {len(r2_ids)} finalists ({current_sims}/{total_sims} sims) | ETA: {eta_secs:.1f}s")
                                        
                                        # Clear the progress bar and restore the UI opacity when complete!
                                        synth_progress_ui.empty()
                                        
                                        # Auto-Calibrate hardware speed using this deep run
                                        synth_elapsed = time.time() - synth_start_time
                                        if synth_elapsed > 0:
                                            st.session_state.sims_per_sec = max(1, int(total_sims / synth_elapsed))
                                            
                                        for args, r in zip(r2_args, res2):
                                            b_id = args['_b_id']
                                            t_val = float(r.get(run_target_metric, 0.0))
                                            f_val = r.get("highest_floor", 0)
                                            
                                            build_res[b_id]['sum_t'] += t_val
                                            build_res[b_id]['sum_f'] += f_val
                                            build_res[b_id]['floors'].append(f_val)
                                            
                                        # SORTING LOGIC FOR ROUND 2
                                        if run_target_metric == "highest_floor":
                                            best_b_id = sorted(r2_ids, key=lambda k: get_ceiling_score(build_res[k]['floors'], 5), reverse=True)[0]
                                        else:
                                            best_b_id = sorted(r2_ids, key=lambda k: build_res[k]['sum_t'], reverse=True)[0]
                                            
                                        best_data = build_res[best_b_id]
                                        final_meta_dist = dict(best_b_id)
                                        
                                        abs_max = max(best_data['floors'])
                                        avg_f = best_data['sum_f'] / 500.0
                                        
                                        synth_summary = {
                                            run_target_metric: abs_max if run_target_metric == "highest_floor" else best_data['sum_t'] / 500.0,
                                            "avg_floor": avg_f,
                                            "abs_max_floor": abs_max,
                                            "abs_max_chance": best_data['floors'].count(abs_max) / 500.0,
                                            "floors": best_data['floors'],
                                            "worst_val": 0,
                                            "avg_val": avg_f,
                                            "runner_up_val": 0,
                                            # Dynamically populate the target metric so the main UI Analytics tabs wake up!
                                            "avg_metrics": {run_target_metric: best_data['sum_t'] / 500.0} 
                                        }

                                        # APPLES-TO-APPLES CHART MAPPING
                                        same_target_runs =[]
                                        for r in valid_runs:
                                            b_id = tuple({s: r[s] for s in stat_keys}.items())
                                            if run_target_metric == "highest_floor":
                                                # CEILING SCORE: Absolute peaks are skewed by 1-in-a-million RNG. 
                                                # We chart the Top 5 Peak Average to perfectly match the Engine's sorting logic.
                                                ceiling = get_ceiling_score(build_res[b_id]['floors'], 5)
                                                same_target_runs.append(ceiling)
                                            else:
                                                # CONSISTENCY: Averages must be strictly regressed to the mean via 500 runs.
                                                same_target_runs.append(build_res[b_id]['sum_t'] / 500.0)
                                                
                                        if run_target_metric == "highest_floor":
                                            meta_score = get_ceiling_score(best_data['floors'], 5)
                                            chart_label = "🏆 Theoretical Peak"
                                        else:
                                            meta_score = best_data['sum_t'] / 500.0
                                            chart_label = "📈 Optimal Farm-Build"
                                            
                                        avg_history_score = sum(same_target_runs)/len(same_target_runs) if same_target_runs else 0.0
                                        
                                        st.session_state.opt_results["best_final"] = final_meta_dist
                                        st.session_state.opt_results["final_summary_out"] = synth_summary
                                        st.session_state.opt_results["chart_hill_labels"] =[chart_label, "🧬 Polished Meta-Build"]
                                        st.session_state.opt_results["chart_hill_scores"] =[avg_history_score, meta_score]
                                        st.session_state.opt_results["chart_hist"] = dict(Counter(best_data['floors']))
                                        
                                        abs_max_chance = best_data['floors'].count(abs_max) / 500.0
                                        
                                        # Calculate exact stamina cost using the isolated Meta-Build stats
                                        import copy
                                        temp_p = copy.deepcopy(p)
                                        for k, v in final_meta_dist.items(): temp_p.base_stats[k] = v
                                        arch_secs_cost = math.ceil(1.0 / abs_max_chance) * temp_p.max_sta if abs_max_chance > 0 else 0
                                        
                                        # Save locally with telemetry so we can chart it below the button!
                                        st.session_state.synthesis_result = {
                                            "stats": final_meta_dist,
                                            "meta_score": meta_score,
                                            "history_scores": same_target_runs,
                                            "metric_name": run_target_metric,
                                            "abs_max": abs_max,
                                            "abs_max_chance": abs_max_chance,
                                            "arch_secs_cost": arch_secs_cost
                                        }
                                        
                                        # --- APPEND TO SYNTHESIS HISTORY ---
                                        if "synth_history" not in st.session_state:
                                            st.session_state.synth_history =[]
                                            
                                        synth_entry = {
                                            "Target": run_target_metric,
                                            "Ceiling Score": round(meta_score, 2),
                                            "Sources Data": valid_runs # Save the full dictionaries for the sub-table!
                                        }
                                        if run_target_metric == "highest_floor":
                                            synth_entry["Theoretical Peak"] = int(abs_max)
                                            synth_entry["Peak Probability"] = abs_max_chance
                                            synth_entry["Arch Secs Cost"] = arch_secs_cost
                                            
                                        synth_entry.update(final_meta_dist)
                                        st.session_state.synth_history.append(synth_entry)
                                        
                                        st.rerun()

                        with col_synth2:
                            if st.button("🗑️ Delete Unchecked Runs", width="stretch", help="Permanently deletes any visible runs that do NOT have their 'Include' box checked."):
                                # 1. Preserve runs that are currently hidden by the target filter
                                hidden_runs =[r for r in st.session_state.run_history if r.get("Target") not in view_targets]
                                
                                # 2. Preserve only the visible runs that the user left CHECKED
                                kept_visible_runs =[r for r in visible_history if r.get("Include", False)]
                                        
                                # 3. Overwrite history (Unchecked runs are dropped into the void)
                                st.session_state.run_history = hidden_runs + kept_visible_runs
                                
                                # Flush data editor memory to prevent shape mismatch errors
                                for k in list(st.session_state.keys()):
                                    if k.startswith("history_editor_"):
                                        del st.session_state[k]
                                        
                                st.toast("🗑️ Unchecked runs permanently deleted!", icon="🧹")
                                st.rerun()
                                
                        st.divider()
                        st.markdown("#### 📋 Run History Table")
                        st.write("*(Check the **Include** box for your top 2-5 runs to mix them into your Meta-Build. You can permanently **delete** unchecked runs using the trash can button above!)*")
                        
                        df_history = pd.DataFrame(visible_history)

                        # --- Inject Card Reality Check Columns for Farming ---
                        is_block_farming = any("block_" in t for t in view_targets)
                        if is_block_farming:
                            def get_50_str(score, odds):
                                if score <= 0: return "N/A"
                                return f"~{(0.693 * odds) / (score / 60.0) / 1000.0:.1f}k"
                            
                            df_history["Base Card (50%)"] = df_history.apply(lambda row: get_50_str(row["Metric Score"], 1500) if "block_" in row.get("Target", "") else "-", axis=1)
                            df_history["Poly (50%)"] = df_history.apply(lambda row: get_50_str(row["Metric Score"], 7500) if "block_" in row.get("Target", "") else "-", axis=1)
                            df_history["Infernal (50%)"] = df_history.apply(lambda row: get_50_str(row["Metric Score"], 200000) if "block_" in row.get("Target", "") else "-", axis=1)
                        
                        # Dynamically name the score column based on what the user is viewing
                        is_only_floor = all(t == "highest_floor" for t in view_targets)
                        score_col = "Score (Floor)" if is_only_floor else "Yield (1k Arch Secs)"

                        # Transform the display column for non-floor targets securely without breaking the backend
                        df_history[score_col] = df_history.apply(
                            lambda row: row["Metric Score"] if row.get("Target") == "highest_floor" else round((row["Metric Score"] / 60.0) * 1000.0, 1), 
                            axis=1
                        )
                        
                        cols =[ 'Include', 'Target', score_col ]
                        if is_block_farming:
                            cols +=[ "Base Card (50%)", "Poly (50%)", "Infernal (50%)" ]
                        cols +=[ 'Avg Floor', 'Max Floor' ]
                        
                        # Safe fallback in case old history rows don't have Max Floor yet
                        if 'Max Floor' not in df_history.columns: df_history['Max Floor'] = 0 
                        cols +=[ c for c in df_history.columns if c not in cols and c != "_global_idx" and c != "Metric Score" ]
                        df_history = df_history[cols]
                        
                        # Create a robust, unique key to anchor the frontend state
                        view_targets_str = "_".join(view_targets)
                        editor_key = f"history_editor_{len(visible_history)}_{view_targets_str}"
                        
                        def on_history_change():
                            """Callback to map frontend edits securely to the backend BEFORE the script reruns."""
                            if editor_key not in st.session_state: return
                            edits = st.session_state[editor_key].get("edited_rows", {})
                            for row_idx_str, edit_dict in edits.items():
                                if "Include" in edit_dict:
                                    row_idx = int(row_idx_str)
                                    # Use the closure's visible_history from the previous render to map the index
                                    global_idx = visible_history[row_idx]["_global_idx"]
                                    st.session_state.run_history[global_idx]["Include"] = edit_dict["Include"]

                        edited_df = st.data_editor(
                            df_history, 
                            hide_index=True, 
                            width="stretch",
                            column_config={"Include": st.column_config.CheckboxColumn("Include")},
                            disabled=[c for c in df_history.columns if c != "Include"],
                            key=editor_key,
                            on_change=on_history_change
                        )
                                
                        # --- RENDER SYNTHESIS RESULT DIRECTLY IN TAB ---
                        if "synthesis_result" in st.session_state:
                            sr = st.session_state.synthesis_result
                            
                            # State migration failsafe: clear old format if it persists in RAM
                            if "history_scores" not in sr:
                                del st.session_state["synthesis_result"]
                                st.rerun()
                                
                            st.success("✅ Synthesis Complete! The main Advanced Analytics charts on the **Optimizer** tab have been updated to reflect this new Meta-Build.")
                            
                            # --- 📊 PERFORMANCE PROOF CHART ---
                            st.markdown("#### 📊 Synthesis Performance Proof")
                            st.write("How the optimized Meta-Build compares to the individual historical runs you selected.")
                            st.caption("*(Note: To ensure a mathematically fair comparison, your historical runs were re-evaluated alongside the new combinations using the same 500-simulation baseline to remove RNG variance).*")
                            
                            chart_labels =[f"Run {i+1}" for i in range(len(sr["history_scores"]))] +["🧬 Meta-Build"]
                            
                            is_floor_target = (sr.get("metric_name", "highest_floor") == "highest_floor")
                            def scale_sr(v): return v if is_floor_target else (v / 60.0) * 1000.0
                            
                            chart_scores = [scale_sr(s) for s in sr["history_scores"]] +[scale_sr(sr["meta_score"])]
                            chart_colors = ["Historical Runs"] * len(sr["history_scores"]) + ["Meta-Build"]
                            
                            df_comp = pd.DataFrame({"Build": chart_labels, "Score": chart_scores, "Type": chart_colors})
                            
                            # Dynamically zoom the Y-axis so fractional improvements on plateaus are highly visible
                            min_score = min(chart_scores) * 0.98 if chart_scores else 0
                            
                            fig_comp = px.bar(df_comp, x="Build", y="Score", color="Type", text_auto='.3s',
                                              color_discrete_map={"Historical Runs": "#6495ED", "Meta-Build": "#4CAF50"})
                            fig_comp.update_layout(showlegend=False, margin=dict(t=10, b=20), height=300)
                            fig_comp.update_yaxes(range=[min_score, max(chart_scores) * 1.02])
                            st.plotly_chart(fig_comp, width="stretch")
                            
                            # --- 🏆 META-BUILD YIELDS & CALCULATOR ---
                            st.divider()
                            m_name = sr.get("metric_name", "highest_floor")
                            m_score = sr.get("meta_score", 0)
                            
                            if m_name == "highest_floor":
                                c_m1, c_m2 = st.columns(2)
                                c_m1.metric("🏔️ Top 5 Peak Average (Ceiling)", f"Floor {m_score:,.1f}")
                                c_m2.metric("🏆 Theoretical Peak (1-in-500)", f"Floor {sr.get('abs_max', m_score):,.0f}")
                                
                                # --- 🎲 PEAK REALITY CHECK ---
                                chance = sr.get('abs_max_chance', 0)
                                if chance > 0:
                                    runs_needed = math.ceil(1.0 / chance)
                                    arch_secs = sr.get('arch_secs_cost', runs_needed * p.max_sta)
                                    st.info(f"🎲 **Peak Reality Check:** The AI hit Floor {sr.get('abs_max')} in **{chance*100:.1f}%** of its simulations. Mathematically, you must execute an average of **{runs_needed} full runs** to see this happen once. At your current Max Stamina, expect to burn roughly **~{arch_secs/1000.0:.1f}k Arch Seconds** before you break through! *(If you spent less than this and didn't hit it, you just haven't banked enough arch seconds yet!)*")
                            else:
                                m_str = "Fragments" if "frag" in m_name else "Kills" if "block" in m_name else "EXP"
                                r_1k = (m_score / 60.0) * 1000.0
                                
                                c_m1, c_m2 = st.columns(2)
                                c_m1.metric(f"💰 {m_str} per 1k Arch Secs", f"{r_1k:,.1f}")
                                c_m2.metric(f"⏱️ {m_str} per minute", f"{m_score:,.2f}")
                                
                                if m_name == "xp_per_min":
                                    st.markdown("##### ⬆️ Level Up Calculator")
                                    col_sx_c, col_sx_t = st.columns(2)
                                    with col_sx_c:
                                        s_cur_xp = st.number_input("Current EXP", min_value=0.0, step=1000.0, format="%.0f", key="synth_cur_xp")
                                    with col_sx_t:
                                        s_tar_xp = st.number_input("Target EXP", min_value=0.0, step=1000.0, format="%.0f", key="synth_tar_xp")
                                        
                                    if s_cur_xp > 0 or s_tar_xp > 0:
                                        if s_tar_xp > s_cur_xp and m_score > 0:
                                            s_mins = (s_tar_xp - s_cur_xp) / m_score
                                            st.info(f"**Required:** ~{(s_mins * 60.0) / 1000.0:,.1f}k Arch Seconds ({s_mins:,.1f} mins real-time)")
                                        else:
                                            st.warning("Target EXP must be greater than Current EXP.")
                                elif "block_" in m_name and m_score > 0:
                                    st.markdown("##### 🎴 Card Drop Reality Check (Arch Secs)")
                                    b_name = m_name.replace("block_", "").replace("_per_min", "").capitalize()
                                    
                                    def calc_c_main(odds):
                                        k50 = (0.693 * odds) / (m_score / 60.0) / 1000.0
                                        k90 = (2.302 * odds) / (m_score / 60.0) / 1000.0
                                        k99 = (4.605 * odds) / (m_score / 60.0) / 1000.0
                                        return f"~{k50:.1f}k / ~{k90:.1f}k / ~{k99:.1f}k"
                                        
                                    st.info(f"**{b_name} Card Projections[50% Average / 90% Safe / 99% Guaranteed]**\n\n"
                                            f"**Base Card:** {calc_c_main(1500)} &nbsp;&nbsp;|&nbsp;&nbsp; "
                                            f"**Poly Frag:** {calc_c_main(7500)} &nbsp;&nbsp;|&nbsp;&nbsp; "
                                            f"**Infernal Frag:** {calc_c_main(200000)}")
                                            
                            st.divider()
                            
                            # --- 🧬 STAT OUTPUT ---
                            st.markdown("#### 🧬 Synthesized Stat Allocation")
                            
                            synth_stat_cols = st.columns(len(sr["stats"]))
                            for idx_s, (stat_name, allocated_pts) in enumerate(sr["stats"].items()):
                                with synth_stat_cols[idx_s]:
                                    with st.container(border=True):
                                        img_path = os.path.join(ROOT_DIR, "assets", "stats", f"{stat_name.lower()}.png")
                                        if os.path.exists(img_path):
                                            render_centered_image(img_path, 250) 
                                        else:
                                            st.markdown(f"<div style='text-align:center;'><b>{stat_name}</b></div>", unsafe_allow_html=True)
                                        
                                        current_val = int(st.session_state.get(f"stat_{stat_name}", p.base_stats.get(stat_name, 0)))
                                        delta = int(allocated_pts) - current_val
                                        st.metric(label=stat_name, value=int(allocated_pts), delta=delta, label_visibility="collapsed")
                            
                            col_ma1, col_ma2 = st.columns(2)
                            with col_ma1:
                                st.button("✨ Apply Meta-Build Globally", width="stretch", key="apply_meta_build_btn", on_click=cb_apply_stats, args=("global", sr["stats"], "✅ Meta-Build stats applied globally!", "🧬"))
                            with col_ma2:
                                st.button("🧪 Send Meta-Build to Sandbox", width="stretch", key="sandbox_meta_build_btn", on_click=cb_apply_stats, args=("sandbox", sr["stats"], "✅ Meta-Build piped to Tab 6 (Hit Calculator)!", "🧪"))

            # ==========================================
                # META-BUILD HISTORY TABLE (NESTED EXPANDERS)
                # ==========================================
                if "synth_history" in st.session_state and st.session_state.synth_history:
                    st.divider()
                    st.markdown("### 📚 Meta-Build History Log")
                    st.write("A permanent record of your optimized Meta-Builds. Expand a row to view the original builds that birthed it.")
                    
                    synth_targets = list(set(s.get("Target") for s in st.session_state.synth_history))
                    synth_view_targets = st.multiselect(
                        "🔍 Filter Meta-Builds by target:", 
                        options=synth_targets, 
                        default=[t for t in synth_targets if t == run_target_metric] or synth_targets,
                        key="synth_filter_ms"
                    )
                    
                    # Iterate backwards so the newest Meta-Builds are always at the top!
                    for idx, synth in reversed(list(enumerate(st.session_state.synth_history))):
                        if synth.get("Target") in synth_view_targets:
                            
                            with st.container(border=True):
                                # --- Header & Visible Stats ---
                                is_floor_target = (synth.get('Target', 'highest_floor') == 'highest_floor')
                                disp_score = synth['Ceiling Score'] if is_floor_target else round((synth['Ceiling Score'] / 60.0) * 1000.0, 1)
                                
                                title = f"#### 🧬 Meta-Build | Target: `{synth['Target']}` | Ceiling: `{disp_score}`"
                                if not is_floor_target: title += " *(per 1k Arch Secs)*"
                                
                                if "Theoretical Peak" in synth: 
                                    title += f" | Peak: `{synth['Theoretical Peak']}`"
                                elif "God-Run Peak" in synth: # Legacy state fallback
                                    title += f" | Peak: `{synth['God-Run Peak']}`"
                                st.markdown(title)
                                
                                stats_only = {k: v for k, v in synth.items() if k not in["Target", "Ceiling Score", "Theoretical Peak", "Peak Probability", "God-Run Peak", "God-Run Chance", "Arch Secs Cost", "Sources Data", "Sources", "Keep"]}
                                stat_string = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join([f"**{k}:** {v}" for k, v in stats_only.items()])
                                st.info(stat_string)
                                
                                chance = synth.get("Peak Probability", synth.get("God-Run Chance", 0))
                                if chance > 0:
                                    runs_needed = math.ceil(1.0 / chance)
                                    arch_secs = synth.get("Arch Secs Cost", 0)
                                    peak_val = synth.get('Theoretical Peak', synth.get('God-Run Peak'))
                                    st.caption(f"🎲 **Reality Check:** Floor {peak_val} hit in **{chance*100:.1f}%** of sims. Requires avg **{runs_needed} runs** (~**{arch_secs/1000.0:.1f}k Arch Secs**) to replicate.")
                                    
                                if "block_" in synth['Target'] and synth['Ceiling Score'] > 0:
                                    val = synth['Ceiling Score']
                                    b_name = synth['Target'].replace("block_", "").replace("_per_min", "").capitalize()
                                    
                                    def calc_c(odds):
                                        k50 = (0.693 * odds) / (val / 60.0) / 1000.0
                                        k90 = (2.302 * odds) / (val / 60.0) / 1000.0
                                        k99 = (4.605 * odds) / (val / 60.0) / 1000.0
                                        return f"~{k50:.1f}k / ~{k90:.1f}k / ~{k99:.1f}k"
                                        
                                    st.caption(f"🎴 **Card Reality Check ({b_name})**[50% Avg / 90% Safe / 99% Guaranteed] ➔ "
                                               f"**Base:** {calc_c(1500)} &nbsp;|&nbsp; "
                                               f"**Poly:** {calc_c(7500)} &nbsp;|&nbsp; "
                                               f"**Infernal:** {calc_c(200000)}", unsafe_allow_html=True)
                                
                                # --- Hidden Source Runs ---
                                with st.expander("🔍 View Source Runs (The original builds used to generate this Meta-Build)"):
                                    if "Sources Data" in synth:
                                        source_df = pd.DataFrame(synth['Sources Data'])
                                        
                                        # Apply the same dynamic formatting as the main history table
                                        is_synth_floor = (synth.get('Target') == "highest_floor")
                                        score_col = "Score (Floor)" if is_synth_floor else "Yield (1k Arch Secs)"
                                        
                                        source_df[score_col] = source_df.apply(
                                            lambda row: row.get("Metric Score") if is_synth_floor else round((row.get("Metric Score", 0) / 60.0) * 1000.0, 1), 
                                            axis=1
                                        )
                                        
                                        cols_to_drop =['Include', 'Target', 'Metric Score', '_global_idx'] 
                                        source_df = source_df.drop(columns=[c for c in cols_to_drop if c in source_df.columns])
                                        
                                        # Reorder to put the new score column at the front
                                        s_cols = [score_col] +[c for c in source_df.columns if c != score_col]
                                        source_df = source_df[s_cols]
                                        
                                        st.dataframe(source_df, hide_index=True, width="stretch")
                                    else:
                                        st.write(synth.get("Sources", "*(No source data saved)*"))
                                
                                # --- Always-Visible Buttons ---
                                col_h1, col_h2, col_h3 = st.columns(3)
                                
                                col_h1.button("✨ Apply Globally", key=f"app_hist_{idx}", width="stretch", on_click=cb_apply_stats, args=("global", stats_only, "✅ Meta-Build stats applied globally!", "🧬"))
                                    
                                col_h2.button("🧪 Send to Sandbox", key=f"snd_hist_{idx}", width="stretch", on_click=cb_apply_stats, args=("sandbox", stats_only, "✅ Meta-Build piped to Tab 6 (Hit Calculator)!", "🧪"))
                                    
                                col_h3.button("🗑️ Delete Meta-Build", key=f"del_hist_{idx}", width="stretch", on_click=cb_delete_hist, args=(idx,))

            # ==========================================
            # NEXT STEPS: ROI ANALYZER (TAB ROUTING)
            # ==========================================
            with tab_res_roi:
                st.markdown("### 🔮 Upgrade Guide (Marginal ROI)")
                
                if run_target_metric == "highest_floor":
                    st.warning("⚠️ **ROI Analyzer is Disabled for Max Floor Push:**\nBecause floor progression relies on large, discrete math 'Breakpoints' (e.g., shaving a 3-hit kill down to a 2-hit kill), adding a single +1 to a stat rarely shows an immediate gain. Additionally, the ROI engine compares a 15-run average to your absolute Peak God Run, which mathematically causes false negatives.\n\nTo calculate exactly what stats you need to beat your current wall, send your build to **Tab 6 (Hit Calculator Sandbox)** and manually inspect the HP and Armor Breakpoints!")
                else:
                    st.markdown("Wondering what to buy next? The ROI Analyzer runs isolated micro-simulations, adding **+1 Level** to every stat and un-maxed upgrade, then ranks them by their immediate raw boost to your yields.")
                    
                    st.warning("⚠️ **Note:** This engine ranks **raw output gain**, not cost efficiency. You must weigh the AI's top recommendations against your actual in-game fragment costs!")
                    
                    with st.expander("🤓 Deep Dive: Why are some of my results negative?"):
                        st.markdown("""
                        If your top upgrades are negative, it means no remaining upgrades significantly help your current goal. This happens for two reasons:
                        1. **The Suicide Farming Paradox:** If you are farming early-game drops (e.g. Dirt cards), buying survival upgrades (Stamina) pushes you into deeper floors where blocks have exponentially more HP. Fighting deep-floor blocks takes longer, which mathematically *lowers* your early-game kills/minute!
                        2. **Statistical Noise:** If an upgrade provides absolutely 0 benefit to your goal, the natural RNG variance of a rapid 15-run micro-test will cause it to fluctuate slightly into the negatives.
                        """)
                    
                    st.divider()
                    
                    col_roi_1, col_roi_2 = st.columns(2)
                    
                    with col_roi_1:
                        st.markdown("##### 1. Next Stat Point")
                        st.write("Tests adding +1 to every stat to see which yields the highest increase.")
                        
                        if st.button("🔍 Analyze Next Stat Point", width="stretch"):
                            with st.spinner("Testing marginal stat values..."):
                                stat_results = {}
                                
                                roi_state_dict = {
                                    'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                                    'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                                    'asc1_unlocked': p.asc1_unlocked, 'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                                    'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                                    'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
                                    'total_infernal_cards': p.total_infernal_cards
                                }
                                
                                roi_pool_args =[]
                                cap_increase = int(p.u('H45'))
                                
                                for s in best_final.keys():
                                    max_cap = cfg.BASE_STAT_CAPS.get(s, 0) + cap_increase
                                    if best_final[s] < max_cap:
                                        test_dist = best_final.copy()
                                        test_dist[s] += 1
                                        for _ in range(15):
                                            roi_pool_args.append({
                                                'stats': test_dist, 'fixed_stats': {}, 'state_dict': roi_state_dict, '_test_stat': s
                                            })
                                            
                                if sys.platform == "linux": CPU_CORES = min(2, mp.cpu_count()) 
                                else: CPU_CORES = max(1, mp.cpu_count() - 1)
                                
                                if roi_pool_args:
                                    with mp.Pool(CPU_CORES) as pool:
                                        res_list = pool.map(worker_simulate, roi_pool_args)
                                        
                                        for args, r in zip(roi_pool_args, res_list):
                                            t_s = args['_test_stat']
                                            if t_s not in stat_results: stat_results[t_s] = {'sum': 0, 'count': 0}
                                            val = float(r.get(run_target_metric, 0.0))
                                            stat_results[t_s]['sum'] += val
                                            stat_results[t_s]['count'] += 1
                                            
                                    base_val = final_summary_out.get(run_target_metric, 0)
                                    st.session_state.roi_stat_results = {
                                        k: (((v['sum']/v['count']) - base_val) / 60.0) * 1000.0 
                                        for k, v in stat_results.items()
                                    }
                                    st.rerun() 
                                else:
                                    st.warning("All stats are already maxed out! No further points can be tested.")
                                    
                        if "roi_stat_results" in st.session_state:
                            sorted_stats = sorted(st.session_state.roi_stat_results.items(), key=lambda x: x[1], reverse=True)
                            df_stat_roi = pd.DataFrame(sorted_stats, columns=["Stat (+1)", "Marginal Gain (1k Arch Secs)"])
                            st.dataframe(df_stat_roi, hide_index=True, width="stretch")
    
                    with col_roi_2:
                        st.markdown("##### 2. Upgrade ROI (Internal)")
                        st.write("Tests adding +1 level to every un-maxed internal upgrade.")
                        
                        if st.button("🔍 Analyze Upgrades", width="stretch"):
                            with st.spinner("Testing marginal upgrade values (This may take a minute)..."):
                                upg_results = {}
                                roi_pool_args =[]
                                
                                for upg_id, upg_data in p.UPGRADE_DEF.items():
                                    current_lvl = p.upgrade_levels.get(upg_id, 0)
                                    max_lvl = cfg.INTERNAL_UPGRADE_CAPS.get(upg_id, 99)
                                    
                                    asc2_locked_rows =[19, 27, 34, 46, 52, 55]
                                    if not p.asc2_unlocked and upg_id in asc2_locked_rows: continue
                                        
                                    if current_lvl < max_lvl:
                                        roi_state_dict = {
                                            'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                                            'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                                            'asc1_unlocked': p.asc1_unlocked, 'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                                            'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                                            'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
                                            'total_infernal_cards': p.total_infernal_cards
                                        }
                                        roi_state_dict['upgrade_levels'][upg_id] = current_lvl + 1
                                        
                                        for _ in range(15):
                                            roi_pool_args.append({
                                                'stats': best_final, 'fixed_stats': {}, 'state_dict': roi_state_dict, '_test_upg': upg_data[0]
                                            })
                                            
                                if sys.platform == "linux": CPU_CORES = min(2, mp.cpu_count()) 
                                else: CPU_CORES = max(1, mp.cpu_count() - 1)
                                
                                if roi_pool_args:
                                    with mp.Pool(CPU_CORES) as pool:
                                        res_list = pool.map(worker_simulate, roi_pool_args)
                                        
                                        for args, r in zip(roi_pool_args, res_list):
                                            t_u = args['_test_upg']
                                            if t_u not in upg_results: upg_results[t_u] = {'sum': 0, 'count': 0}
                                            val = float(r.get(run_target_metric, 0.0))
                                            upg_results[t_u]['sum'] += val
                                            upg_results[t_u]['count'] += 1
                                            
                                    base_val = final_summary_out.get(run_target_metric, 0)
                                    st.session_state.roi_upg_results = {
                                        k: (((v['sum']/v['count']) - base_val) / 60.0) * 1000.0 
                                        for k, v in upg_results.items()
                                    }
                                    st.rerun() 
                                else:
                                    st.warning("All internal upgrades are maxed out! No further upgrades can be tested.")
                                    
                        if "roi_upg_results" in st.session_state:
                            sorted_upgs = sorted(st.session_state.roi_upg_results.items(), key=lambda x: x[1], reverse=True)
                            df_upg_roi = pd.DataFrame(sorted_upgs[:10], columns=["Upgrade (+1 Lvl)", "Marginal Gain (1k Arch Secs)"])
                            st.dataframe(df_upg_roi, hide_index=True, width="stretch")

    # --- GLOBAL FLOATING NAVIGATION ---
    st.markdown('<a href="#top-of-tabs" class="back-to-top">⬆️ Back to Tabs</a>', unsafe_allow_html=True)