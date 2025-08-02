from pathlib import Path

import pandas as pd

# Get project root: parent of src/
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_FILE = PROJECT_ROOT / "data/raw/New York CitiBike - 2015-2017.csv"
Q_SOURCE_FILE = PROJECT_ROOT / "data/processed/Q Table - Source.csv"
Q_TARGET_FILE = PROJECT_ROOT / "data/processed/Q Table - Target.csv"
TRAINING_REWARDS = PROJECT_ROOT / "results/rewards/SmartFlow - Rewards.csv"
MAP_OUTPUT = PROJECT_ROOT / "results/simulation/SmartFlow - Simulation.html"

TARGET_DATE = pd.to_datetime("2016-07-01").date()
TOP_N = 10
LOG_INTERVAL = 500

EPISODES = 10000
MAX_STEPS = 100  # up to 100 moves per episode
ALPHA = 0.3
GAMMA = 0.9  # reward discount
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995  # slow decay → more exploration

MAP_CENTER = [40.72274243859797, -74.06340830643403]
MAP_ZOOM = 14

OSRM_URL = (
    "http://router.project-osrm.org/route/v1/driving/"
    "{from_lon},{from_lat};{to_lon},{to_lat}"
    "?overview=full&geometries=geojson"
)

BIKE_ICON_URL = "https://cdn-icons-png.flaticon.com/512/684/684908.png"
TRUCK_ICON_URL = "https://cdn-icons-png.flaticon.com/512/1995/1995471.png"
