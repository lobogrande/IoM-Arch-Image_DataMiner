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
from optimizers.parallel_worker import run_optimization_phase, benchmark_hardware, get_eta_profiles, worker_simulate

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


    # ==========================================
    # MAIN WINDOW: Tabs
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

    tab_stats, tab_upgrades, tab_cards, tab_calc_stats, tab_block_stats, tab_sandbox, tab_optimizer = st.tabs([
        "📊 Base Stats", "⬆️ Upgrades", "🎴 Block Cards", "📋 Calculated Player Stats", "🪨 Block Stats", "🧪 Block Hit Sandbox", "🚀 Run Optimizer"
    ])

    # --- TAB 1: BASE STATS ---
    with tab_stats:
        
        # --- ACTIONABLE EMPTY STATE & PRESETS ---
        if "opt_results" not in st.session_state:
            with st.container(border=True):
                st.markdown("### 👋 Welcome to the Optimizer!")
                st.write("If you are new here, follow these 3 steps to get started:")
                st.markdown("1. **Input your Stats & Upgrades:** Use the first 3 tabs above, import your own json player data, or click a **Preset Build** to auto-fill realistic data.\n2. **Select your Goal:** Go to the **Run Optimizer** tab and choose your target.\n3. **Run the Engine:** Let the Monte Carlo simulations find your perfect mathematical build.")
                
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
                        early_game = {"settings": {"asc2_unlocked": False, "arch_level": 45, "current_max_floor": 40, "base_damage_const": 10, "total_infernal_cards": 0, "arch_ability_infernal_bonus": 0.0}, "base_stats": {"Str": 15, "Agi": 0, "Per": 0, "Int": 0, "Luck": 20, "Div": 10}, "internal_upgrades": {"3 - Gem Stamina": 25, "4 - Gem Exp": 12, "5 - Gem Loot": 12, "9 - Flat Damage": 15, "10 - Armor Pen.": 15, "11 - Exp. Gain": 15, "12 - Stat Points": 3, "13 - Crit Chance/Damage": 12, "14 - Max Sta/Sta Mod Chance": 12, "15 - Flat Damage": 8, "16 - Loot Mod Gain": 6, "17 - Unlock Fairy/Armor Pen": 6, "18 - Enrage&Crit Dmg/Enrage Cooldown": 5, "20 - Flat Dmg/Super Crit Chance": 5, "21 - Exp Gain/Fragment Gain": 4, "22 - Flurry Sta Gain/Flurr Cooldown": 4, "23 - Max Sta/Sta Mod Gain": 4, "24 - All Mod Chances": 3, "25 - Flat Dmg/Damage Up": 0, "26 - Max Sta/Mod Chance": 0, "28 - Exp Gain/Max Sta": 3, "29 - Armor Pen/Ability Cooldowns": 3, "30 - Crit Dmg/Super Crit Dmg": 3, "31 - Quake Atks/Cooldown": 3, "32 - Flat Dmg/Enrage Cooldown": 0, "33 - Mod Chance/Armor Pen": 0, "35 - Exp Gain/Mod Ch.": 0, "36 - Damage Up/Armor Pen": 0, "37 - Super Crit/Ultra Crit Chance": 0, "38 - Exp Mod Gain/Chance": 0, "39 - Ability Insta Chance/Max Sta": 0, "40 - Ultra Crit Dmg/Sta Mod Chance": 0, "41 - Poly Card Bonus": 0, "42 - Frag Gain Mult": 0, "43 - Sta Mod Gain": 0, "44 - All Mod Chances": 0, "45 - Exp Gain/All Stat Cap Inc.": 0, "47 - Damage Up/Crit Dmg Up": 0, "48 - Gold Crosshair Chance/Auto-Tap Chance": 0, "49 - Flat Dmg/Ultra Crit Chance": 0, "50 - Ability Insta Chance/Sta Mod Chance": 0, "51 - Dmg Up/Exp Gain": 0, "53 - Super Crit Dmg/Exp Mod Gain": 0, "54 - Max Sta/Crosshair Auto-Tap Chance": 0}, "external_upgrades": {"Hestia Idol": 0, "Axolotl Skin": 9, "Dino Skin": 9, "Geoduck Tribute": 750, "Avada Keda- Skill": 1, "Block Bonker Skill": 1, "Archaeology Bundle": 0, "Ascension Bundle": 0, "Arch Ability Card": 3}, "cards": {"dirt1": 3, "dirt2": 2, "dirt3": 2, "com1": 3, "com2": 2, "com3": 2, "rare1": 3, "rare2": 2, "rare3": 2, "epic1": 2, "epic2": 2, "epic3": 2, "leg1": 2, "leg2": 2, "leg3": 2, "myth1": 2, "myth2": 2, "myth3": 2, "div1": 2, "div2": 0, "div3": 0}}
                        apply_preset(early_game)
                        
                with col_p2:
                    if st.button("🌌 Load Late-Game Build\n(Asc 2, Floor 158)", width="stretch"):
                        late_game = {"settings": {"asc2_unlocked": True, "arch_level": 99, "current_max_floor": 158, "base_damage_const": 10, "hades_idol_level": 129, "total_infernal_cards": 303, "arch_ability_infernal_bonus": -0.1509}, "base_stats": {"Str": 15, "Agi": 0, "Per": 0, "Int": 29, "Luck": 30, "Div": 15, "Corr": 15}, "internal_upgrades": {"3 - Gem Stamina": 50, "4 - Gem Exp": 25, "5 - Gem Loot": 25, "9 - Flat Damage": 25, "10 - Armor Pen.": 25, "11 - Exp. Gain": 25, "12 - Stat Points": 5, "13 - Crit Chance/Damage": 25, "14 - Max Sta/Sta Mod Chance": 20, "15 - Flat Damage": 20, "16 - Loot Mod Gain": 10, "17 - Unlock Fairy/Armor Pen": 15, "18 - Enrage&Crit Dmg/Enrage Cooldown": 15, "19 - Gleaming Floor Chance": 30, "20 - Flat Dmg/Super Crit Chance": 25, "21 - Exp Gain/Fragment Gain": 20, "22 - Flurry Sta Gain/Flurr Cooldown": 10, "23 - Max Sta/Sta Mod Gain": 5, "24 - All Mod Chances": 30, "25 - Flat Dmg/Damage Up": 5, "26 - Max Sta/Mod Chance": 5, "27 - Unlock Ability Fairy/Loot Mod Gain": 20, "28 - Exp Gain/Max Sta": 15, "29 - Armor Pen/Ability Cooldowns": 10, "30 - Crit Dmg/Super Crit Dmg": 20, "31 - Quake Atks/Cooldown": 10, "32 - Flat Dmg/Enrage Cooldown": 5, "33 - Mod Chance/Armor Pen": 5, "34 - Buff Divinity[Div Stats Up]": 5, "35 - Exp Gain/Mod Ch.": 5, "36 - Damage Up/Armor Pen": 20, "37 - Super Crit/Ultra Crit Chance": 20, "38 - Exp Mod Gain/Chance": 20, "39 - Ability Insta Chance/Max Sta": 20, "40 - Ultra Crit Dmg/Sta Mod Chance": 20, "41 - Poly Card Bonus": 1, "42 - Frag Gain Mult": 1, "43 - Sta Mod Gain": 1, "44 - All Mod Chances": 1, "45 - Exp Gain/All Stat Cap Inc.": 1, "46 - Gleaming Floor Multi": 24, "47 - Damage Up/Crit Dmg Up": 1, "48 - Gold Crosshair Chance/Auto-Tap Chance": 5, "49 - Flat Dmg/Ultra Crit Chance": 5, "50 - Ability Insta Chance/Sta Mod Chance": 25, "51 - Dmg Up/Exp Gain": 5, "52 - [Corruption Buff] Dmg Up / Mod Multi Up": 10, "53 - Super Crit Dmg/Exp Mod Gain": 30, "54 - Max Sta/Crosshair Auto-Tap Chance": 28, "55 - All Mod Multipliers": 10}, "external_upgrades": {"Hestia Idol": 1929, "Axolotl Skin": 11, "Dino Skin": 11, "Geoduck Tribute": 1047, "Avada Keda- Skill": 1, "Block Bonker Skill": 1, "Archaeology Bundle": 1, "Ascension Bundle": 1, "Arch Ability Card": 4}, "cards": {"dirt1": 4, "dirt2": 4, "dirt3": 4, "dirt4": 3, "com1": 3, "com2": 3, "com3": 4, "com4": 2, "rare1": 3, "rare2": 3, "rare3": 3, "rare4": 2, "epic1": 3, "epic2": 3, "epic3": 4, "epic4": 2, "leg1": 3, "leg2": 3, "leg3": 4, "leg4": 2, "myth1": 3, "myth2": 3, "myth3": 3, "myth4": 2, "div1": 3, "div2": 3, "div3": 3, "div4": 0}}
                        apply_preset(late_game)
                        
                with col_p3:
                    if st.button("🗑️ Factory Reset\n(Wipe All Data)", width="stretch", type="secondary"):
                        apply_preset(reset=True)
                        
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
            # --- HIDE MAXED TOGGLE ---
            hide_maxed = st.toggle("👀 Hide Maxed Upgrades", value=False)
            st.divider()

            asc2_locked_rows =[19, 27, 34, 46, 52, 55]
            
            # 1. Pre-filter active upgrades
            active_upgrades = list()
            for upg_id, upg_data in p.UPGRADE_DEF.items():
                if not p.asc2_unlocked and upg_id in asc2_locked_rows:
                    continue
                    
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
            # --- TOTAL INFERNAL CARDS INPUT ---
            col_inf_tog, _ = st.columns([1, 2])
            with col_inf_tog:
                if "set_total_inf" not in st.session_state:
                    st.session_state["set_total_inf"] = int(p.total_infernal_cards)
                    
                p.total_infernal_cards = st.number_input(
                    "Total Infernal Cards (Global)", 
                    min_value=0, step=1, key="set_total_inf",
                    help="Sum of all Infernal cards you own across all categories (Archaeology, Fishing, etc). Used for the Infernal Multiplier."
                )
            st.divider()

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
                    "Damage": {"stats": ["Str", "Corr", "Div"], "upgs":[9, 15, 20, 25, 32, 34, 36, 47, 49, 51, 52], "exts":["Dino Skin", "Hestia Idol"]},
                    "Armor Pen": {"stats": ["Per", "Int"], "upgs": [10, 17, 29, 33, 36], "exts": []},
                    "Max Stamina": {"stats": ["Agi", "Corr"], "upgs":[3, 14, 23, 26, 28, 39, 54], "exts": []},
                    "Crit Chances & Multipliers": {"stats":["Luck", "Div"], "upgs":[13, 18, 20, 30, 37, 40, 47, 49, 53], "exts":[]},
                    "EXP & Fragment Gain": {"stats": ["Int", "Per", "Div"], "upgs":[4, 11, 21, 28, 35, 42, 45, 51], "exts": ["Axolotl Skin", "Geoduck Tribute"]},
                    "Mod Chances & Multipliers": {"stats": ["Luck", "Div", "Corr"], "upgs":[5, 14, 16, 23, 24, 26, 33, 35, 38, 40, 43, 44, 48, 50, 52, 53, 54, 55], "exts": ["Archaeology Bundle"]},
                    "Abilities (Instacharge / Cooldowns)": {"stats": ["Int", "Div"], "upgs":[18, 22, 29, 31, 32, 39, 50], "exts":["Arch Ability Card", "Avada Keda- Skill", "Block Bonker Skill"]}
                }
                
                data = TROUBLESHOOT_MAP[troubleshoot_stat]
                
                # Dynamically extract all live external upgrade values
                ext_vals = {}
                for group in cfg.EXTERNAL_UI_GROUPS:
                    ext_vals[group['name']] = int(p.external_levels.get(group['rows'][0], 0))
                
                t_col1, t_col2, t_col3 = st.columns(3)
                
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
                if st.button("🔄 Sync from Base Stats", width="stretch", help="Pull your currently saved stat distribution into the sandbox."):
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
        st.write("Leverage Successive Halving to find the absolute mathematically perfect stat distribution. Ensure your total allocated points do not exceed your budget before running.")

        # --- PROJECTION DISCLAIMER ---
        with st.expander("⚠️ Important Disclaimer regarding Projections (Click to read)"):
            disclaimer_text = (
                "**⚠️ IMPORTANT DISCLAIMER REGARDING PROJECTIONS:**\n\n"
                "**The Good News:** The environment generation in this engine is now **100% identical** to the live game's source code! "
                "The stat distributions this tool provides are mathematically perfect for your current upgrades.\n\n"
                "**The Reality Check #1:** While the combat math is exact, the absolute output numbers (Max Floor, Kills/hr) are built on **Statistical Averages**. "
                "The AI runs hundreds of simulations and optimizes for *consistent, reliable farming*. Because it smooths out extreme RNG, "
                "the engine maintains a slightly conservative slant. You may occasionally experience a 'God Run' in the actual game that pushes you "
                "a few floors higher than the AI predicts. Treat these numbers as your highly accurate, reliable baseline!"
                "**The Reality Check #2:** The engine calculates **100% Theoretical Efficiency**. In the Python simulator, 0.000 seconds pass between killing an ore and hitting the next one. "
                "In the actual live game, minor animation delays, frame drops, and tick-rate transitions consume fractions of a second. "
                "Because of this 'Animation Lag', you should expect your actual real-world Yields (XP/Frags) to be roughly **~5% to 10% lower** than the mathematical perfection projected here."
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
            st.info("💡 **Strategy Tip:** Pushing deep floors requires balancing Damage, Armor Pen, Max Stamina and Crits. To make the AI run much faster, try opening the **Stat Constraints** below and locking **Intelligence** to `0` and **Luck** to your max stat cap!")
        elif opt_goal in["Fragment Farming", "Block Card Farming", "Max EXP Yield"]:
            st.info("💡 **Strategy Tip:** If your target spawns on early floors (e.g., Dirt), you don't need Max Stamina or Armor Pen to reach it! Lock **Agility** and **Perception** to `0` to speed up the AI.\n\n⚠️ **Wait, what if my target is late-game?** If you are farming Tier 4 blocks (which spawn on Floor 81+), you STILL have to survive the gauntlet of tough ores to get there. Do not lock your survival stats to 0, or the AI will die before reaching your target!")
        st.divider()

        # --- NEW: STAT LOCKING ---
        with st.expander("🔒 Stat Constraints / Locking (Optional)", expanded=False):
            st.write("Locking a stat to a specific value drastically reduces the multidimensional search space, resulting in much faster simulations. Points used here share your global budget.")
            
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
            render_lock_stat("Divine", 'Div', lcol2)
            if p.asc2_unlocked:
                render_lock_stat("Corruption", 'Corr', lcol3)

        st.divider()

        # --- HARDWARE BENCHMARKING & ETA ---
        if "sims_per_sec" not in st.session_state:
            st.session_state.sims_per_sec = 0

        # --- LIVE ETA RECALCULATION ---
        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
        if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
        
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
                
        live_eta_profiles = get_eta_profiles(STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, eta_bounds, st.session_state.sims_per_sec)

        with st.expander("⚙️ Engine Tuning & Hardware Benchmark (Optional)", expanded=False):
            
            # Keep the interactive controls at the absolute top
            col_bench, col_prof = st.columns([1, 1.5])
            
            with col_bench:
                st.write("#### 1. Hardware Benchmark")
                st.write("*(Optional: Runs automatically on start if skipped)*")
                if st.button("⏱️ Benchmark CPU & Calculate ETAs", width="stretch"):
                    with st.spinner("Running 200 micro-simulations to test CPU speed..."):
                        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                        if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                        
                        base_state_dict = {
                            'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                            'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                            'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                            'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                            'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
                            'total_infernal_cards': p.total_infernal_cards
                        }
                        
                        # --- GUARANTEED STRESS-TEST BENCHMARK ---
                        # We must test a "Glass Cannon" (High Str/Agi). If the user left their UI on 0 stats,
                        # the benchmark will run instantly and provide a fake 10,000 sims/sec speed.
                        bench_budget = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
                        bench_stats = {s: 0 for s in STATS_TO_OPTIMIZE}
                        
                        if bench_budget > 0:
                            bench_stats['Str'] = min(99, bench_budget)
                            if 'Agi' in bench_stats:
                                bench_stats['Agi'] = max(0, bench_budget - bench_stats['Str'])
                                
                        payload = {'stats': bench_stats, 'fixed_stats': {}, 'state_dict': base_state_dict}
                        
                        if sys.platform == "linux":
                            CPU_CORES = min(2, mp.cpu_count()) 
                        else:
                            CPU_CORES = max(1, mp.cpu_count() - 1)
                            
                        with mp.Pool(CPU_CORES) as pool:
                            spd = benchmark_hardware(payload, pool)
                            st.session_state.sims_per_sec = spd
                            st.rerun() 
                
                if st.session_state.sims_per_sec > 0:
                    st.success(f"⚡ **Hardware Speed:** {st.session_state.sims_per_sec:,.0f} simulations / second")
                else:
                    st.info("Awaiting Benchmark...")

            with col_prof:
                st.write("#### 2. Search Depth (Initial Step Size)")
                
                depth_labels = {
                    "Fast": "Fast (Step 15) - Best for quick checks",
                    "Standard": "Standard (Step 10) - Recommended balance",
                    "Deep": "Deep (Step 5) - Exhaustive, takes much longer"
                }
                
                depth_choice = st.radio(
                    "Select Search Depth", 
                    options=list(depth_labels.keys()), 
                    index=1,
                    format_func=lambda x: depth_labels[x],
                    horizontal=False, 
                    label_visibility="collapsed"
                )

                st.divider()
                st.write("#### 3. Execution Time Limit")
                time_limit_mins = st.slider(
                    "Safely abort and return best build if time exceeds:", 
                    min_value=1, max_value=30, value=5, step=1, format="%d mins",
                    help="This is a 'Graceful Timeout'. To prevent data corruption, the engine will finish its currently active batch of math before stopping. Expect the final timer to overshoot your limit slightly!"
                )
                
                step_1 = {"Fast": 15, "Standard": 10, "Deep": 5}[depth_choice]
                step_2 = max(2, step_1 // 3)
                step_3 = 1
                
                preview_html = f"""
                <div style='font-size: 0.9em; padding: 10px; border-left: 3px solid #4CAF50; background-color: rgba(76, 175, 80, 0.1); margin-top: 10px;'>
                    <b>Engine Execution Plan:</b><br>
                    🔍 <b>Phase 1:</b> Scanning grid in leaps of <b>{step_1}</b>...<br>
                    🔎 <b>Phase 2:</b> Zooming in with leaps of <b>{step_2}</b>...<br>
                    🎯 <b>Phase 3:</b> Pinpointing exact peak with leaps of <b>{step_3}</b>.
                """
                
                if st.session_state.sims_per_sec > 0:
                    prof_key = next(k for k in live_eta_profiles.keys() if k.startswith(depth_choice))
                    prof_data = live_eta_profiles[prof_key]
                    preview_html += f"<br><br>⏱️ <b>Estimated Time:</b> {prof_data['time_label']} <i>(~{prof_data['builds']:,.0f} unique builds tested)</i>"
                else:
                    preview_html += "<br><br>⏱️ <b>Estimated Time:</b> Awaiting Benchmark..."
                    
                preview_html += "</div>"
                st.markdown(preview_html, unsafe_allow_html=True)

            # --- UN-INDENTED EXPLANATION TEXT ---
            st.divider()
            st.markdown("""
            **🧠 How does the AI Optimizer work?**
            Testing every stat combination point-by-point would take days. Instead, we "zoom in":
            * **Phase 1 (Coarse):** Casts a wide net across your stat budget in large leaps.
            * **Phase 2 (Fine):** Draws a tight box around the Phase 1 winner and tests smaller leaps.
            * **Phase 3 (Exact):** Pinpoints the mathematical peak by testing every single point in that final box.
            
            *(The engine also uses "Successive Halving" to quickly delete bad builds after testing them briefly, saving enormous amounts of time).*
            """)

    # --- MONTE CARLO EXECUTION LOOP ---
        st.divider()
        
        # Hidden for Production Beta. Change to True if you need to do UI testing later!
        # dev_mode = st.toggle("🛠️ UI Dev Mode (Instantly mock results to design UI without running engine)")
        dev_mode = False
        
        # --- ACTIVE SETTINGS TRANSPARENCY ---
        st.info(f"⚙️ **Active Settings:** {depth_choice} Search Depth | {time_limit_mins} Min Timeout. *(Adjust these in the Engine Tuning expander above)*")
        
        # --- PRE-FLIGHT CHECK ---
        # Calculate total locked points to prevent mathematically impossible runs
        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
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

        st.caption("⚠️ **Note:** Do not change tabs or click other widgets while the engine is running, or it will abort the simulation!")
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
                    'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
                    'current_max_floor': p.current_max_floor, 'hades_idol_level': p.hades_idol_level,
                    'arch_ability_infernal_bonus': p.arch_ability_infernal_bonus,
                    'total_infernal_cards': p.total_infernal_cards
                }

                # AUTO-BENCHMARK FAILSAFE
                if st.session_state.sims_per_sec == 0:
                    with st.spinner("⏱️ First-time setup: Benchmarking your CPU..."):
                        STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
                        if p.asc2_unlocked: STATS_TO_OPTIMIZE.append('Corr')
                        
                        bench_budget = int(p.arch_level) + int(p.upgrade_levels.get(12, 0))
                        bench_stats = {s: 0 for s in STATS_TO_OPTIMIZE}
                    
                        # Dump budget into Damage and Stamina to ensure it reaches deep floors
                        if bench_budget > 0:
                            bench_stats['Str'] = min(99, bench_budget)
                            if 'Agi' in bench_stats:
                                bench_stats['Agi'] = max(0, bench_budget - bench_stats['Str'])

                        payload = {'stats': bench_stats, 'fixed_stats': {}, 'state_dict': base_state_dict}
                        # Cloud OOM Protection: Streamlit Linux containers only have 1GB RAM
                        if sys.platform == "linux":
                            CPU_CORES = min(2, mp.cpu_count()) 
                        else:
                            CPU_CORES = max(1, mp.cpu_count() - 1)
                        with mp.Pool(CPU_CORES) as pool:
                            spd = benchmark_hardware(payload, pool)
                            st.session_state.sims_per_sec = spd
                            # Recalculate live profiles now that we have a speed
                            live_eta_profiles = get_eta_profiles(STATS_TO_OPTIMIZE, DYNAMIC_BUDGET, eta_bounds, spd)

                prof_key = next((k for k in live_eta_profiles.keys() if k.startswith(depth_choice)), None)
                if prof_key:
                    prof_data = live_eta_profiles[prof_key]
                    step_size = prof_data['step']
                    st.info(f"⏱️ **Running {depth_choice} Search:** Estimated to take {prof_data['time_label']} (~{prof_data['builds']:,.0f} builds at {st.session_state.sims_per_sec:,.0f} sims/sec)")
                else:
                    step_size = {"Fast": 15, "Standard": 10, "Deep": 5}[depth_choice]
                    st.info(f"⏱️ **Running {depth_choice} Search:** Building optimization grid...")

                with st.spinner(f"Engine Running..."):
                    start_time = time.time()
                    STATS_TO_OPTIMIZE =['Str', 'Agi', 'Per', 'Int', 'Luck', 'Div']
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
                    def st_progress_callback(phase_name, r_idx, r_total, task_idx, task_total):
                        pct = min(100, max(0, int((task_idx / task_total) * 100)))
                        elapsed_now = time.time() - start_time
                        ui_prog_bar.progress(pct, text=f"⚙️ {phase_name} | Round {r_idx}/{r_total} | {pct}% ({task_idx}/{task_total} sims) | ⏱️ Elapsed: {elapsed_now:.1f}s / {time_limit_mins}m limit")
                    
                    time_limit_secs = time_limit_mins * 60
                    
                    import traceback
                    try:
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
                                
                            if best_p2 and (time.time() - start_time) < time_limit_secs:
                                bounds_p3 = {}
                                p3_radius = min(2, step_2) 
                                for s in STATS_TO_OPTIMIZE:
                                    if st.session_state.get(f"lock_check_{s}", False):
                                        bounds_p3[s] = bounds_p1[s]
                                    else:
                                        bounds_p3[s] = (max(0, best_p2[s] - p3_radius), min(EFFECTIVE_CAPS[s], best_p2[s] + p3_radius))
                                        
                                best_p3, final_summary = run_optimization_phase(
                                    "Phase 3 (Exact)", target_metric, STATS_TO_OPTIMIZE, 
                                    DYNAMIC_BUDGET, 1, ITER_P3, pool, FIXED_STATS, bounds_p3,
                                    progress_callback=st_progress_callback, global_start_time=start_time, time_limit_seconds=time_limit_secs,
                                    base_state_dict=base_state_dict
                                )
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
                st.success(f"✅ Successive Halving Complete in {elapsed:.1f} seconds!")
            
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

            st.divider()

            # ==========================================
            # ADVANCED ANALYTICS DASHBOARD (TABS)
            # ==========================================
            st.markdown("### 📊 Advanced Analytics Dashboard")
            tab_list =["📈 Performance"]
            
            if run_target_metric != "highest_floor" or dev_mode: tab_list.append("🃏 Card Drops")
            if show_loot: tab_list.append("🎒 Loot Breakdown")
            if show_wall: tab_list.append("🧱 The Wall")
            
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
                        
                        st.metric("Absolute Max Floor (God Run)", f"Floor {abs_max:,.0f}")
                        st.metric("God Run Probability", f"{abs_chance:.1f}%")
                        st.metric("Average Consistency Floor", f"Floor {avg_flr:,.1f}")
                    else:
                        val = final_summary_out[run_target_metric]
                        rate_1k = (val / 60.0) * 1000.0
                        metric_str = "Fragments" if "frag" in run_target_metric else "Kills" if "block" in run_target_metric else "EXP"
                        
                        # Clean, consolidated Banked Time readout
                        st.markdown(f"#### 💰 Projected Yield<br><span style='font-size: 0.9em; color: gray;'>Target {metric_str} per 1k Arch Seconds</span>", unsafe_allow_html=True)
                        st.metric("Yield", f"{rate_1k:,.1f}", label_visibility="collapsed")
                        
                        st.divider()
                        
                        # Clean, consolidated Real-Time readout
                        # Clean, consolidated Real-Time readout
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
                                    st.success(f"**Required:** ~{(mins_req * 60.0) / 1000.0:,.1f}k Banked Arch Seconds ({mins_req:,.1f} mins real-time)")
                                elif tar_xp <= cur_xp:
                                    st.warning("Target EXP must be greater than Current EXP.")

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
                    st.plotly_chart(fig_hill, width="stretch")
                    
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
                        "Performance":[worst_val, avg_val, runner_up_val, final_summary_out[run_target_metric]]
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
                    st.plotly_chart(fig_loot, width="stretch")

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
            # RUN HISTORY & SYNTHESIS (OUTSIDE TABS)
            # ==========================================
            st.divider()
            with st.container():
                st.markdown("### 📚 Run History & Hybrid Synthesis")
                st.write("Because the combat math is highly balanced, optimizations often land on a 'Plateau' where wildly different builds perform identically. Track your runs here to spot these patterns.")
                
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
                        if st.button("🔄 Toggle 'Include' for Visible", width="stretch"):
                            for r in st.session_state.run_history:
                                if r.get("Target") in view_targets:
                                    r["Include"] = not r.get("Include", True)
                            st.rerun()

                    visible_history =[r for r in st.session_state.run_history if r.get("Target") in view_targets]
                    
                    if not visible_history:
                        st.info("No runs match the selected filters. Run the optimizer to build history.")
                    else:
                        st.write("*(Check the **Include** box to mix runs into your Meta-Build. You can permanently **delete** unchecked runs using the trash can button below!)*")
                        
                        df_history = pd.DataFrame(visible_history)
                        cols =['Include', 'Target', 'Metric Score', 'Avg Floor', 'Max Floor']
                        # Safe fallback in case old history rows don't have Max Floor yet
                        if 'Max Floor' not in df_history.columns: df_history['Max Floor'] = 0 
                        cols +=[c for c in df_history.columns if c not in cols]
                        df_history = df_history[cols]
                        
                        edited_df = st.data_editor(
                            df_history, 
                            hide_index=True, 
                            width="stretch",
                            column_config={"Include": st.column_config.CheckboxColumn("Include")},
                            disabled=[c for c in df_history.columns if c != "Include"] 
                        )
                        
                        st.divider()
                        st.markdown("#### 🧬 Synthesize Meta-Build (Pass 2)")
                        st.info("""
💡 **Strategy Tip: The "Stat Plateau" (Why do my stats change on re-runs?)**

You might notice that running Synthesis multiple times gives slightly different stat numbers. Don't panic—the AI isn't guessing!

* **The Math:** Enemies only take whole hits. If 50 Strength kills a boss in exactly 3 hits, having 54 Strength *also* kills it in 3 hits. This creates a "Stat Plateau" where several different builds are functionally identical and mathematically tied for 1st place.
* **The Tie-Breaker (RNG):** To break the tie, the AI forces these top builds to race 500 times. Whichever tied build happens to get slightly luckier with Critical Hits during that specific race wins the gold medal!

**The Takeaway:** If your stats bounce around slightly between runs, congratulations—you've reached the absolute peak! Send your results to the **Hit Calculator Sandbox** to prove to yourself that both builds kill your target blocks in the exact same number of hits.
                        """)
                        st.write("Smooth out Monte Carlo RNG noise. This algorithm averages your checked builds, corrects for budget constraints, and runs a deep verification test against your *current* UI target.")
                        
                        col_synth1, col_synth2 = st.columns(2)
                        with col_synth1:
                            if st.button("🧬 Synthesize & Verify Meta-Build", width="stretch"):
                                # WYSIWYG Guard: Only synthesize runs that are currently visible in the UI filter!
                                valid_runs = [r for r in visible_history if r.get("Include", False)]
                                
                                if len(valid_runs) == 0:
                                    st.error("⚠️ You must have at least 1 visible run checked to synthesize!")
                                elif len(valid_runs) > 10:
                                    st.error("⚠️ **Safety Limit Reached:** Synthesizing creates dozens of mathematical permutations for every input build. Please select 10 or fewer builds to prevent server memory overloads!")
                                else:
                                    with st.spinner("Calculating seed, mapping neighborhood, and running deep Gradient Polish..."):
                                        stat_keys =[k for k in valid_runs[0].keys() if k not in["Include", "Target", "Metric Score", "Avg Floor", "Max Floor"]]
                                        
                                        synth_state_dict = {
                                            'base_stats': p.base_stats.copy(), 'upgrade_levels': p.upgrade_levels.copy(),
                                            'external_levels': p.external_levels.copy(), 'cards': p.cards.copy(),
                                            'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
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
                                        
                                        synth_prog = st.progress(0, text=f"🧬 Prepping {len(candidates)} build permutations... (~{total_sims:,} sims | ETA: {eta_secs:.1f}s)")
                                        
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
                                        
                                        # Clear the progress bar when complete!
                                        synth_prog.empty()
                                            
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
                                            chart_label = "🏆 Verified God-Build"
                                        else:
                                            meta_score = best_data['sum_t'] / 500.0
                                            chart_label = "📈 Verified Farm-Build"
                                            
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
                                            synth_entry["God-Run Peak"] = int(abs_max)
                                            synth_entry["God-Run Chance"] = abs_max_chance
                                            synth_entry["Arch Secs Cost"] = arch_secs_cost
                                            
                                        synth_entry.update(final_meta_dist)
                                        st.session_state.synth_history.append(synth_entry)
                                        
                                        st.rerun()

                        with col_synth2:
                            if st.button("🗑️ Delete Unchecked Runs", width="stretch", help="Permanently deletes any visible runs that do NOT have their 'Include' box checked."):
                                # 1. Preserve runs that are currently hidden by the target filter
                                hidden_runs =[r for r in st.session_state.run_history if r.get("Target") not in view_targets]
                                
                                # 2. Preserve only the visible runs that the user left CHECKED
                                kept_visible_runs =[]
                                for i, row in edited_df.iterrows():
                                    if row["Include"]:
                                        kept_visible_runs.append(visible_history[i])
                                        
                                # 3. Overwrite history (Unchecked runs are dropped into the void)
                                st.session_state.run_history = hidden_runs + kept_visible_runs
                                st.toast("🗑️ Unchecked runs permanently deleted!", icon="🧹")
                                st.rerun()
                                
                        # --- RENDER SYNTHESIS RESULT DIRECTLY IN TAB ---
                        if "synthesis_result" in st.session_state:
                            sr = st.session_state.synthesis_result
                            
                            # State migration failsafe: clear old format if it persists in RAM
                            if "history_scores" not in sr:
                                del st.session_state["synthesis_result"]
                                st.rerun()
                                
                            st.success("✅ Synthesis Complete! The main charts above have been updated to reflect this build.")
                            
                            # --- 📊 PERFORMANCE PROOF CHART ---
                            st.markdown("#### 📊 Synthesis Performance Proof")
                            st.write("How the Gradient-Polished Meta-Build compares to the individual historical runs you selected.")
                            st.caption("*(Note: To ensure a mathematically fair comparison, your historical runs were re-evaluated to 500 simulations to strip away their initial 'lucky' RNG variance).*")
                            
                            chart_labels =[f"Run {i+1}" for i in range(len(sr["history_scores"]))] + ["🧬 Meta-Build"]
                            chart_scores = sr["history_scores"] + [sr["meta_score"]]
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
                                c_m2.metric("🏆 Absolute God-Run (1-in-500)", f"Floor {sr.get('abs_max', m_score):,.0f}")
                                
                                # --- 🎲 GOD-RUN REALITY CHECK ---
                                chance = sr.get('abs_max_chance', 0)
                                if chance > 0:
                                    runs_needed = math.ceil(1.0 / chance)
                                    arch_secs = sr.get('arch_secs_cost', runs_needed * p.max_sta)
                                    st.info(f"🎲 **God-Run Reality Check:** The AI hit Floor {sr.get('abs_max')} in **{chance*100:.1f}%** of its simulations. Mathematically, you must execute an average of **{runs_needed} full runs** to see this happen once. At your current Max Stamina, expect to burn roughly **~{arch_secs/1000.0:.1f}k Arch Seconds** before you break through! *(If you spent less than this and didn't hit it, you just haven't banked enough arch seconds yet!)*")
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
                                            st.info(f"**Required:** ~{(s_mins * 60.0) / 1000.0:,.1f}k Banked Arch Seconds ({s_mins:,.1f} mins real-time)")
                                        else:
                                            st.warning("Target EXP must be greater than Current EXP.")
                                            
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
                st.write("A permanent record of your synthesized God-Builds. Expand a row to view the original builds that birthed it.")
                
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
                            title = f"#### 🧬 Meta-Build | Target: `{synth['Target']}` | Ceiling: `{synth['Ceiling Score']}`"
                            if "God-Run Peak" in synth: 
                                title += f" | Peak: `{synth['God-Run Peak']}`"
                            st.markdown(title)
                            
                            stats_only = {k: v for k, v in synth.items() if k not in["Target", "Ceiling Score", "God-Run Peak", "God-Run Chance", "Arch Secs Cost", "Sources Data", "Sources", "Keep"]}
                            stat_string = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join([f"**{k}:** {v}" for k, v in stats_only.items()])
                            st.info(stat_string)
                            
                            if "God-Run Chance" in synth and synth["God-Run Chance"] > 0:
                                chance = synth["God-Run Chance"]
                                runs_needed = math.ceil(1.0 / chance)
                                arch_secs = synth.get("Arch Secs Cost", 0)
                                st.caption(f"🎲 **Reality Check:** Floor {synth.get('God-Run Peak')} hit in **{chance*100:.1f}%** of sims. Requires avg **{runs_needed} runs** (~**{arch_secs/1000.0:.1f}k Arch Secs**) to replicate.")
                                
                            if "block_" in synth['Target'] and synth['Ceiling Score'] > 0:
                                val = synth['Ceiling Score']
                                kills_50 = 0.693 * 1500
                                kills_90 = 2.302 * 1500
                                arch_1k_50 = kills_50 / (val / 60.0) / 1000.0
                                arch_1k_90 = kills_90 / (val / 60.0) / 1000.0
                                b_name = synth['Target'].replace("block_", "").replace("_per_min", "").capitalize()
                                st.caption(f"🎴 **Card Reality Check ({b_name}):** Base Card takes **~{arch_1k_50:.1f}k** Arch Secs (50% lucky) up to **~{arch_1k_90:.1f}k** Arch Secs (90% safe).")
                            
                            # --- Hidden Source Runs ---
                            with st.expander("🔍 View Source Runs (The original builds that were averaged)"):
                                if "Sources Data" in synth:
                                    source_df = pd.DataFrame(synth['Sources Data'])
                                    cols_to_drop = ['Include', 'Target'] 
                                    source_df = source_df.drop(columns=[c for c in cols_to_drop if c in source_df.columns])
                                    st.dataframe(source_df, hide_index=True, width="stretch")
                                else:
                                    st.write(synth.get("Sources", "*(No source data saved)*"))
                            
                            # --- Always-Visible Buttons ---
                            col_h1, col_h2, col_h3 = st.columns(3)
                            
                            col_h1.button("✨ Apply Globally", key=f"app_hist_{idx}", width="stretch", on_click=cb_apply_stats, args=("global", stats_only, "✅ Meta-Build stats applied globally!", "🧬"))
                                
                            col_h2.button("🧪 Send to Sandbox", key=f"snd_hist_{idx}", width="stretch", on_click=cb_apply_stats, args=("sandbox", stats_only, "✅ Meta-Build piped to Tab 6 (Hit Calculator)!", "🧪"))
                                
                            col_h3.button("🗑️ Delete Meta-Build", key=f"del_hist_{idx}", width="stretch", on_click=cb_delete_hist, args=(idx,))

            # ==========================================
            # NEXT STEPS: ROI ANALYZER (OUTSIDE TABS)
            # ==========================================
            st.divider()
            st.markdown("### 🔮 Next Steps: Marginal Value & ROI Analyzer")
            
            if run_target_metric == "highest_floor":
                st.warning("⚠️ **ROI Analyzer is Disabled for Max Floor Push:**\nBecause floor progression relies on large, discrete math 'Breakpoints' (e.g., shaving a 3-hit kill down to a 2-hit kill), adding a single +1 to a stat rarely shows an immediate gain. Additionally, the ROI engine compares a 15-run average to your absolute Peak God Run, which mathematically causes false negatives.\n\nTo calculate exactly what stats you need to beat your current wall, send your build to **Tab 6 (Hit Calculator Sandbox)** and manually inspect the HP and Armor Breakpoints!")
            else:
                st.info("""
💡 **Strategy Tip: Finding your next best upgrade (ROI)**

You just used the Optimizer to find the mathematically perfect build for your *current* stats. But what should you level up *next*?

* **The Micro-Test:** The AI will temporarily add **+1 Level** to every single stat or un-maxed internal upgrade and run a quick batch of simulations.
* **The Ranking:** It then sorts the results to show you exactly which upgrade gives you the biggest immediate raw boost to your Farming Yields (EXP, Fragments, or Cards per minute).

⚠️ **Important Note on Costs:** This engine does *not* currently track the Fragment cost of upgrades. It only measures the **raw output gain**. When using this "shopping list," you must weigh the AI's top recommendations against your actual in-game fragment accumulation rates!
                """)
                st.write("Run isolated micro-simulations to discover exactly where your next investments should go based on your current optimal build.")
                
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
                                'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
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
                                        
                                st.session_state.roi_stat_results = {
                                    k: (v['sum']/v['count']) - final_summary_out.get(run_target_metric, 0) 
                                    for k, v in stat_results.items()
                                }
                                st.rerun() 
                            else:
                                st.warning("All stats are already maxed out! No further points can be tested.")
                                
                    if "roi_stat_results" in st.session_state:
                        sorted_stats = sorted(st.session_state.roi_stat_results.items(), key=lambda x: x[1], reverse=True)
                        df_stat_roi = pd.DataFrame(sorted_stats, columns=["Stat (+1)", "Marginal Gain"])
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
                                        'asc2_unlocked': p.asc2_unlocked, 'arch_level': p.arch_level,
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
                                        
                                st.session_state.roi_upg_results = {
                                    k: (v['sum']/v['count']) - final_summary_out.get(run_target_metric, 0) 
                                    for k, v in upg_results.items()
                                }
                                st.rerun() 
                            else:
                                st.warning("All internal upgrades are maxed out! No further upgrades can be tested.")
                                
                    if "roi_upg_results" in st.session_state:
                        sorted_upgs = sorted(st.session_state.roi_upg_results.items(), key=lambda x: x[1], reverse=True)
                        df_upg_roi = pd.DataFrame(sorted_upgs[:10], columns=["Upgrade (+1 Lvl)", "Marginal Gain"])
                        st.dataframe(df_upg_roi, hide_index=True, width="stretch")

    # --- GLOBAL FLOATING NAVIGATION ---
    st.markdown('<a href="#top-of-tabs" class="back-to-top">⬆️ Back to Tabs</a>', unsafe_allow_html=True)