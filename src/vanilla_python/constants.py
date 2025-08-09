from pathlib import Path

import pandas as pd

# ── Project Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ── Data Input Files ────────────────────────────────────────────────────────────
TRIPS_FILE = PROJECT_ROOT / "data/processed/trips_final.csv"
STATIONS_FILE = PROJECT_ROOT / "data/processed/stations_final.csv"

# ── Data Output Files ───────────────────────────────────────────────────────────
Q_SOURCE_FILE = PROJECT_ROOT / "data/processed/Q Table - Source.csv"
Q_TARGET_FILE = PROJECT_ROOT / "data/processed/Q Table - Target.csv"
TRAINING_REWARDS = PROJECT_ROOT / "results/rewards/SmartFlow - Rewards.csv"
MAP_OUTPUT = PROJECT_ROOT / "results/simulation/SmartFlow - Simulation.html"

# ── Experiment Configuration ────────────────────────────────────────────────────
TARGET_DATE = pd.to_datetime("2016-07-01").date()
TOP_N = 10
LOG_INTERVAL = 500

# ── Q-Learning Hyperparameters ──────────────────────────────────────────────────
EPISODES = 10_000  # total training episodes
MAX_STEPS = 100  # max moves per episode
ALPHA = 0.3  # learning rate
GAMMA = 0.9  # discount factor
EPSILON_START = 1.0  # initial exploration rate
EPSILON_END = 0.01  # final exploration rate
EPSILON_DECAY = 0.9995  # per-episode decay factor

# ── Map & Visualization Settings ───────────────────────────────────────────────
MAP_CENTER = [40.72274243859797, -74.06340830643403]
MAP_ZOOM = 14

# ── External API URLs & Icons ───────────────────────────────────────────────────
OSRM_URL = (
    "http://router.project-osrm.org/route/v1/driving/"
    "{from_lon},{from_lat};{to_lon},{to_lat}"
    "?overview=full&geometries=geojson"
)
BIKE_ICON_URL = "https://cdn-icons-png.flaticon.com/512/684/684908.png"
TRUCK_ICON_URL = "https://cdn-icons-png.flaticon.com/512/1995/1995471.png"

# ── Column Names ────────────────────────────────────────────────────────────────
COL_START_NAME = "Start Station Name"
COL_END_NAME = "End Station Name"
COL_START_LAT = "Start Station Latitude"
COL_START_LON = "Start Station Longitude"
