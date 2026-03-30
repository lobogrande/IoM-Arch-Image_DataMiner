# ⛏️ Idle Obelisk Miner (IoM) Arch Optimizer

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?logo=streamlit)
![Multiprocessing](https://img.shields.io/badge/Engine-Multiprocessing-4CAF50.svg)
![Status](https://img.shields.io/badge/Status-Beta-ffa229.svg)

A high-performance **Monte Carlo Simulator and AI Build Optimizer** for the Archaeology mini-game in *Idle Obelisk Miner*. 

This tool evaluates your player stats, upgrades, and block card collections to compute the absolute perfect stat distribution for your account. Whether you are pushing for a new Max Floor, farming late-game Block Cards, or maximizing EXP yields, this engine mathematically eliminates the guesswork.

---

## 🧠 The Math Problem: Why You Can't "Guess" a Build

Idle Obelisk Miner has a deceptively complex combat engine. Finding the perfect stat distribution manually is nearly impossible due to three mathematical realities encoded into the game:

1. **The Stat Plateau (Truncation):** The game runs in Unity (C#) and casts floats to integers via strict truncation (`math.floor`), not standard rounding. Because blocks only take whole hits, having 50 Strength and 54 Strength might both result in a "3-hit kill". Any stat points spent that do not push you past the next *Breakpoint* are mathematically wasted.
2. **Multiplicative Menus:** In-game stats combine percentage bonuses from different menus multiplicatively, not additively. 
3. **The "Suicide Farming Paradox":** Because the game has zero death-delay, buying survival stats (Agility/Stamina) when farming early-game blocks (e.g., Dirt Cards) pushes the player to deeper floors where block HP is exponentially higher. This causes your kills-per-minute to mathematically *plummet*. 

**The Solution:** This engine emulates the C# source code exactly, executing hundreds of thousands of micro-tick combat simulations to find the optimal breakpoints for your specific target.

---

## 🏗️ Architecture & Stack

The codebase is separated into highly modular architectural layers, decoupling the UI from the heavy mathematical engine to allow for absolute maximum multiprocessing throughput.

* **Layer 1: Configuration (`project_config.py`)** 
  Holds Base Block Stats, Upgrade Caps, and `EXTERNAL_UI_GROUPS`.
* **Layer 2: Core Math (`core/player.py`, `core/block.py`, `core/skills.py`)**
  The mathematical heart. Translates player inputs into exact combat multipliers (Armor Pen, Nested Crits, Ability Instacharge loops). Features native support for Asc1 (Divinity) and Asc2 (Corruption).
* **Layer 3: State Management (`tools/verify_player.py`)**
  Handles JSON importing/exporting with a hybrid-key parser that sanitizes your UI inputs into the core math engine.
* **Layer 4: The Simulation Engine (`engine/floor_map.py` & `engine/combat_loop.py`)**
  A high-speed micro-tick combat simulator. Uses **heavy loop-hoisting** and attribute caching to prevent dictionary lookup overhead. Uses **top-down sequential binomial rolling arrays** (instead of Gaussian guesswork) to mimic the exact in-game block spawn logic.
* **Layer 5: AI Optimizers (`optimizers/parallel_worker.py` + `opt_*.py`)**
  Houses specific targeting scripts (Max Floor, Card Farming, EXP Rate) using a **3-Phase Successive Halving Algorithm**. The AI zooms in: casting a wide net (Phase 1), drawing a tight box around the winner (Phase 2), and calculating the exact point-by-point peak (Phase 3). 
* **Layer 6: The Web Interface (`app.py`)**
  A highly polished Streamlit frontend featuring dynamic React DOM manipulation, custom Base64 HTML image injection, CSS Flexbox centering, and cache-busting Javascript logic.

---

## 🚀 Key Features

* **Hardware-Aware Auto-Scaling:** The engine dynamically adjusts its workload based on your OS context. It detects memory-sharing capabilities (`fork` on Linux vs. `spawn` on Mac/Windows) to calibrate its precision grid instantly, preventing IPC serialization bottlenecks.
* **Deep Tie-Breaker Tournaments:** If two stat builds tie for 1st place, the AI throws them into a 500-iteration Monte Carlo race to see which build performs better against extreme RNG variations.
* **Dynamic Precision Gauge:** A visual UI component that analyzes your Time Limit and Stat Locks, adjusting the "Step Size" leaps the AI takes and warning you if the mathematical net is too wide.
* **Marginal ROI Analyzer:** Evaluates your current character and tests adding `+1` to every possible stat and un-maxed upgrade, ranking them by their raw output gain to tell you exactly what to buy next.
* **Meta-Build Synthesizer:** Merge your historical runs into the ultimate Meta-Build by allowing the engine to calculate the statistical center of your favorite builds and generate nearby hybrid permutations.

---

## 🛠️ Developer Tools included

* **`tools/visualize_run.py`**: A Matplotlib diagnostic dashboard that runs a single combat simulation and outputs a 4-panel graphical analysis of Stamina Depletion, Speed Pool Usage, TTK Pacing, and Hit Distributions.
* **`tools/clean_assets.py`**: A custom Tkinter batch image patcher. Lets developers draw a bounding box over unwanted dynamic text (like in-game numbers) on a template image, seamlessly patching all UI assets in a target directory with the surrounding background color.

---

## 💻 Local Installation & Setup

To run the simulator locally with maximum hardware performance (bypassing cloud memory limits):

**1. Clone the repository**
```bash
git clone https://github.com/lobogrande/IoM-Arch-Image_DataMiner.git
cd IoM-Arch-Image_DataMiner/07_Modeling_and_Simulation
```

**2. Create a virtual environment & install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set up your secrets (Optional, for Feedback Webhook)**
Create a `.streamlit/secrets.toml` file and add your Beta Key and Discord Webhook:
```toml
BETA_KEY = "YourBetaPassword"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/your_webhook_url_here"
```

**4. Launch the application**
```bash
streamlit run app.py
```

---

## 🤝 Acknowledgments & Beta Testers

A massive thank you to the dedicated Discord community members who helped stress-test the math engine, uncover edge cases, and shape the UI into what it is today. 

⭐ **Sans**  
⭐ **Eugloopy☆Dilemma**  
⭐ **Saronitian**  
⭐ **Doctorcool**  
⭐ **Koksuone**  
⭐ **Dustin**  
⭐ **Dave**  

*(If you contributed to the beta testing and your name is missing, please submit a note via the Feedback tab in the app!)*

---

## 📜 Roadmap & Future Plans
**Phase 3:** Once the Python math engine is fully polished and validated by the community, the ultimate goal is to architect a completely Client-Side web app using React / WebAssembly / Pyodide. Executing simulations directly in local browser Web Workers will provide infinite free scalability and completely eliminate server constraints.