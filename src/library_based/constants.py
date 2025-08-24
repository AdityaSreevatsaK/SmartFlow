from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# ── Data Input Files ──────────────────────────────────────────────────────────
TRIPS_FILE = PROJECT_ROOT / "data/processed/trips_final.csv"
STATIONS_FILE = PROJECT_ROOT / "data/processed/stations_final.csv"

# ── Experiment Configuration ──────────────────────────────────────────────────
TARGET_DATE = pd.to_datetime("2016-07-16").date()
TOP_N = 30

# ── RL Environment & Training Hyperparameters ─────────────────────────────────
MAX_STEPS = 100  # Max moves per episode
GAMMA = 0.9  # Discount factor for rewards
TOTAL_TIME_STEPS = 1_000_000
LOG_INTERVAL = 1_000

# ── Inventory model and New York City Graph ─────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "results/models/DQN_Inventory_Model.zip"
NYC_GRAPH_PATH = PROJECT_ROOT / "results/models/NYC_Graph.gpickle"

# ── Map & Visualization Settings ──────────────────────────────────────────────
MAP_CENTER = [40.72274243859797, -74.06340830643403]
MAP_ZOOM = 14

# ── External API URLs & Icons ─────────────────────────────────────────────────
BIKE_ICON_URL = "https://cdn-icons-png.flaticon.com/512/684/684908.png"
TRUCK_ICON_URL = "https://cdn-icons-png.flaticon.com/512/1995/1995471.png"

# ── Column Names ──────────────────────────────────────────────────────────────
COL_START_NAME = "Start Station Name"
COL_END_NAME = "End Station Name"
COL_START_LAT = "Start Station Latitude"
COL_START_LON = "Start Station Longitude"
COL_PRECIP = "precip"
