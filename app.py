# ==============================================================================
# Script: app.py
# Layer 5: Streamlit Web UI (Skeleton)
# Description: The visual frontend for the AI Arch Optimizer. Provides a clean, 
#              tabbed interface for data entry and running Monte Carlo scripts.
# ==============================================================================

import streamlit as st
import json
import os
import sys

# Tell app.py where to find the simulation engine
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SIM_DIR = os.path.join(ROOT_DIR, "07_Modeling_and_Simulation")
if SIM_DIR not in sys.path:
    sys.path.append(SIM_DIR)

# (We will import Player and the optimizers here in the next step!)

# Set page to wide mode for better data display
st.set_page_config(page_title="AI Arch Optimizer", layout="wide", page_icon="⛏️")

# ==========================================
# SIDEBAR: File Management & Global Settings
# ==========================================
with st.sidebar:
    st.header("📂 Player Data")
    st.write("Upload your save file to auto-fill the inputs.")
    
    uploaded_file = st.file_uploader("Upload player_state.json", type=["json"])
    
    st.divider()
    
    st.header("⚙️ Global Settings")
    # These would normally auto-fill from the JSON, but we provide toggles here
    asc2_unlocked = st.checkbox("Ascension 2 Unlocked", value=False)
    arch_level = st.number_input("Arch Level", min_value=1, value=90)
    current_max_floor = st.number_input("Max Floor Reached", min_value=1, value=100)
    
    st.divider()
    st.success("Ready to optimize.")

# ==========================================
# MAIN WINDOW: Tabs
# ==========================================
st.title("⛏️ AI Arch Mining Optimizer")

# Create neat, organized tabs
tab_stats, tab_upgrades, tab_cards, tab_optimizer = st.tabs([
    "📊 Base Stats", 
    "⬆️ Upgrades", 
    "🃏 Cards", 
    "🚀 Run Optimizer"
])

# --- TAB 1: BASE STATS ---
with tab_stats:
    st.subheader("Base Stat Allocation")
    st.info("Upload your JSON on the left, or manually adjust your stats below. Images will go here!")
    
    # Example of how we will do image + input layouts using columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # st.image("assets/stats/str.png", width=50) # Uncomment when you have the image
        st.number_input("Strength (Str)", min_value=0, max_value=55, value=50)
    with col2:
        st.number_input("Agility (Agi)", min_value=0, max_value=55, value=0)
    with col3:
        st.number_input("Perception (Per)", min_value=0, max_value=30, value=0)
    with col4:
        st.number_input("Intelligence (Int)", min_value=0, max_value=30, value=0)

# --- TAB 2: UPGRADES ---
with tab_upgrades:
    st.subheader("Internal & External Upgrades")
    st.write("We will use Streamlit Expanders or Grids here to show icons for all 72 upgrades.")

# --- TAB 3: CARDS ---
with tab_cards:
    st.subheader("Card Collection")
    st.write("A grid of 28 card images with tier-selectors (0 to 4) beneath them.")

# --- TAB 4: OPTIMIZER ---
with tab_optimizer:
    st.subheader("Target Optimization")
    st.write("This is where the user will select what they want to optimize for (Exp/Hr, Max Floor, etc).")
    
    opt_choice = st.selectbox(
        "What is your goal?",["Maximize Exp Yield", "Push Max Floor", "Farm Divinity Fragments", "Farm Specific Cards"]
    )
    
    st.button("Run Simulation", type="primary")