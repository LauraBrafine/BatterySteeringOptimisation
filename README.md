# Battery Steering Optimisation

**Battery Steering Optimisation** is a Python project using [Pyomo](http://www.pyomo.org/) to formulate and solve an MILP that maximises weekly revenue for a 1 MW / 2 MWh battery on the passive imbalance market. Decisions are taken per 15-minute PTU. The default solver is [HiGHS](https://highs.dev).

What the model does (current version):

    - Optimises charge/discharge power each PTU subject to: power limit (|P| ≤ 1 MW),
    - Energy (SoC) bounds (0–2 MWh),
    - Round-trip efficiency,
    - Ramp limits,
    - Daily cycle cap (equivalent full cycles),
    - linear degradation cost per MWh throughput.
    - Handles dual pricing conservatively (implicity in objective function since in all cases of dual pricing short price > long price)
    

Default parameters:

```bash
    # General input parameters
    Delta_t              = 0.25    # hours per period, since each t is a PTU, used to convert power to energy
    MAX_CAPACITY         = 2       # MWh
    MIN_CAPACITY         = 0       # MWh
    MAX_POWER            = 1       # MW for charging and discharging
    INIT_CAPACITY        = 0       # MWh
    EFFICIENCY           = 0.9     # roundtrip efficiency
    RAMP_CH              = 1.0     # MW per PTU for discharging
    RAMP_DC              = 1.0     # MW per PTU for discharging
    CYCLES_PER_DAY_MAX   = 2       # <= 2 equivalent full cycles per day  
    DEG_COST_EUR_PER_MWH = 1.5     # linear degradation €/MWh throughput
```

For a more elaborate problem description, such as mathematical formulation please refer to the docs/ folder or dive into source code for each model in src/bso/ folder.

---

## Installation

```bash
# 1) Clone
git clone https://github.com/yourusername/battery-steering-optimisation.git
cd battery-steering-optimisation

# 2) (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3) Install dependencies (you need python 3.11)
pip install -r requirements.txt
```

---

### HiGHS Solver

This project uses [HiGHS](https://highs.dev) as the optimisation solver. You’ll need to install the HiGHS binaries or library separately so Pyomo can call it.

```bash
# with conda
conda install -c conda-forge pyomo highspy
# with pip
pip install pyomo highspy
```

On Linux/macOS you may also need the system binary `highs` if Pyomo can’t find it:

```bash
# Debian/Ubuntu
sudo apt-get install highs

# macOS (Homebrew)
brew install highs
 ```

Verify installation

```python
from pyomo.environ import SolverFactory

opt = SolverFactory("highs")
print("HiGHS available:", opt.available())
```

---

## Usage

Run from main script (run_optimisation.py) and tweak input params

Or run from CLI:

```bash
# default (v010, highs)
python scripts/run_optimisation.py
```

---

## Project Structure

```text
battery-steering-optimisation/      
├─ data/                                                # Input dataset 
│  └─ imbalance_prices.csv                
├─ docs/                                                # Project documentation
│    ├─ notebooks/                       
│    │  ├─ 01_eda.ipynb                 
│    │  └─ 02_result_visualisation.ipynb 
│    └─ problem-outline/          
│       ├─ Slides_Battery_Steering_Optimisation.pdf              
│       └─ Optimisation_Formulation.png
├─ results/                                             # Model outputs
│  └─ battery_week_results.csv
├─ scripts/                                             # Main entry point to run optimisation
│  └─ run_optimisation.py                  
├─ src/                                                 # Source code for reusable logic
│  └─ bso/                          
│     ├─ __init__.py                
│     ├─ model_v010.py                                  # MILP optimisation model   
│     ├─ test_validation.py                             # tests for model validation      
│     ├─ utils.py                           
│     └─ visualisation.py           
├─ README.md                          
└─ requirements.txt                           
```

---

## Results & docs

- Slides with the approach and findings: docs/problem-outline/Slides_Battery_Steering_Optimisation.pdf.
- Result visualisation: docs/notebooks/02_result_visualisation.ipynb.
- CSV output: results/battery_week_results.csv.

---

## License

Proprietary (for personal use)
