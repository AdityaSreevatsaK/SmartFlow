import logging
import os
import pickle
import random
from typing import Dict

import dotenv
import osmnx as ox
import pandas as pd
from huggingface_hub import login
from shapely.geometry import MultiPoint

from .constants import (COL_END_NAME, COL_START_LAT, COL_START_LON, COL_START_NAME, NYC_GRAPH_PATH, STATIONS_FILE,
                        TRIPS_FILE)

LOG = logging.getLogger(__name__)
dotenv.load_dotenv()


def load_and_filter(date: pd.Timestamp) -> pd.DataFrame:
    """
    Loads trip data from `TRIPS_FILE`, removes rows with missing start station or start time,
    strips whitespace from station names, filters trips to those starting on the specified date,
    and returns the filtered DataFrame.

    Args:
        date (pd.Timestamp): The date to filter trips by.

    Returns:
        pd.DataFrame: Filtered DataFrame of trips for the given date.
    """
    df = pd.read_csv(
        TRIPS_FILE, parse_dates=["Start Time", "Stop Time"], low_memory=False
    )
    df = df.dropna(subset=[COL_START_NAME, "Start Time"])
    df[COL_START_NAME] = df[COL_START_NAME].str.strip()
    df[COL_END_NAME] = df[COL_END_NAME].str.strip()
    df = df[df["Start Time"].dt.date == date]
    LOG.info(f"Loaded & filtered {len(df)} enriched trips on {date}")
    return df


def load_stations() -> pd.DataFrame:
    """
    Loads station metadata from the file specified by `STATIONS_FILE`, strips whitespace from station names,
    renames columns to standardized names, and sets the station name as the DataFrame index.

    Returns:
        pd.DataFrame: DataFrame indexed by station name, containing latitude and longitude columns.
    """
    stations = pd.read_csv(STATIONS_FILE)
    stations["Station Name"] = stations["Station Name"].str.strip()
    stations = stations.rename(
        columns={
            "Station Name": COL_START_NAME, "Latitude": COL_START_LAT, "Longitude": COL_START_LON
        }
    ).set_index(COL_START_NAME)
    return stations


def calculate_thresholds(counts: pd.Series, capacities: Dict[str, int], avg_precip: float = 0.0) -> Dict[str, int]:
    """
    Calculates dynamic rebalancing thresholds for bike stations based on trip counts, station capacities, and average precipitation.

    Args:
        counts (pd.Series): Series mapping station names to trip counts.
        capacities (Dict[str, int]): Dictionary mapping station names to their maximum bike capacities.
        avg_precip (float, optional): Average precipitation value (default is 0.0).

    Returns:
        Dict[str, int]: Dictionary mapping station names to their calculated rebalancing thresholds.
    """
    rain_factor = 1.0 - min(avg_precip / 10.0, 0.2)
    thresholds: Dict[str, int] = {}
    for station, cnt in counts.items():
        base = max(int(cnt / 8 * rain_factor), 3)
        cap = capacities.get(station, base)
        thresholds[station] = min(base, cap)
    return thresholds


def simulate_bike_counts(thresholds: Dict[str, int], capacities: Dict[str, int]) -> Dict[str, int]:
    """
    Simulates random initial bike counts for each station.

    Args:
        thresholds (Dict[str, int]): Dictionary mapping station names to their rebalancing thresholds.
        capacities (Dict[str, int]): Dictionary mapping station names to their maximum bike capacities.

    Returns:
        Dict[str, int]: Dictionary mapping station names to simulated initial bike counts,
                        where each count is a random integer between 0 and the lesser of twice the threshold or the station's capacity.
    """
    return {
        st: random.randint(0, min(thresholds[st] * 2, capacities.get(st, thresholds[st] * 2)))
        for st in thresholds
    }


def preprocess_demand_data(trips_df: pd.DataFrame) -> dict:
    """
    Analyzes trip data to calculate the average hourly demand for each station.

    This function computes the average number of departures and arrivals per hour for each station
    based on the provided trip data. It returns a nested dictionary where each station maps to
    a dictionary of hours (0-23), and each hour contains the average departures and arrivals.

    Args:
        trips_df (pd.DataFrame): DataFrame containing trip records with columns for start time,
                                 end time, start station, and end station.

    Returns:
        dict: Nested dictionary of the form
              {station: {hour: {'departures': float, 'arrivals': float}}}
              representing average hourly demand for each station.
    """
    print("   - Pre-processing trip data to model hourly demand...")
    if 'hour' not in trips_df.columns and 'Start Time' in trips_df.columns:
        trips_df['hour'] = pd.to_datetime(trips_df['Start Time']).dt.hour

    num_days = trips_df['Start Time'].dt.date.nunique()
    departures = trips_df.groupby([COL_START_NAME, 'hour']).size().reset_index(name='trip_count')
    departures['avg_departures'] = departures['trip_count'] / num_days

    arrivals = trips_df.groupby([COL_END_NAME, 'hour']).size().reset_index(name='trip_count')
    arrivals['avg_arrivals'] = arrivals['trip_count'] / num_days

    demand_lookup = {}
    for _, row in departures.iterrows():
        station, hour, avg_deps = row[COL_START_NAME], row['hour'], row['avg_departures']
        if station not in demand_lookup:
            demand_lookup[station] = {h: {'departures': 0, 'arrivals': 0} for h in range(24)}
        demand_lookup[station][hour]['departures'] = avg_deps

    for _, row in arrivals.iterrows():
        station, hour, avg_arrs = row[COL_END_NAME], row['hour'], row['avg_arrivals']
        if station not in demand_lookup:
            demand_lookup[station] = {h: {'departures': 0, 'arrivals': 0} for h in range(24)}
        demand_lookup[station][hour]['arrivals'] = avg_arrs

    print("   - ✅ Demand model created.")
    return demand_lookup


def build_or_load_nyc_graph(coords_top, file_path=NYC_GRAPH_PATH):
    """
    Loads the NYC road network graph from a file if it exists, otherwise builds it from station coordinates and saves it.
    Args:
        coords_top (dict): Dictionary mapping station names to (latitude, longitude) tuples.
        file_path (str, optional): Path to the file for loading/saving the graph. Defaults to `SmartFlow/results/models/nyc_graph.gpickle`.

    Returns:
        networkx.MultiDiGraph: The NYC road network graph.
    """
    if os.path.exists(file_path):
        print(f"✅ Found existing graph. Loading from {file_path}...")
        with open(file_path, "rb") as f:
            G = pickle.load(f)
        print("   - Graph loaded successfully.")
        return G
    else:
        print(f"⚠️ No graph file found. Building new graph...")
        pts = MultiPoint([(lon, lat) for lat, lon in coords_top.values()])
        poly = pts.convex_hull.buffer(0.002)
        G = ox.graph_from_polygon(poly, network_type="drive")
        G = ox.distance.add_edge_lengths(G)
        print(f"   - [TRAINING GRAPH] built with {len(G.nodes)} nodes")
        with open(file_path, "wb") as f:
            pickle.dump(G, f)
        abs_path = os.path.abspath(file_path)
        print(f"   - Training graph saved to {abs_path}")
        return G


def login_to_huggingface():
    """
    Securely logs into Hugging Face using the token stored in the environment variable `HF_TOKEN`.

    This function retrieves the Hugging Face token from the environment, attempts to log in using the `huggingface_hub` library,
    and prints status messages indicating success or failure.

    Raises:
        Exception: If login fails, prints the error message.
    """
    try:
        hf_token = os.getenv("HF_TOKEN")
        print("Logging into Hugging Face...")
        login(token=hf_token)
        print("✅ Successfully logged in.")
    except Exception as e:
        print(f"⚠️ Could not log in. Error: {e}")


def sanitize_for_json(obj):
    """
    Recursively sanitizes Python objects for JSON serialization.
    This function converts dictionaries, lists, NumPy types, arrays, and datetime objects
    into types compatible with JSON serialization. It handles nested structures and
    ensures that all values are converted to standard Python types.

    Args:
        obj: The object to sanitize (can be dict, list, NumPy types, ndarray, datetime, etc.).
    Returns:
        The sanitized object, ready for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(elem) for elem in obj]
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Add handling for datetime objects
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    else:
        return obj


def set_seed(seed_value):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.

    This function initializes the random seed for the Python `random` module, NumPy, and PyTorch (including CUDA if available).
    It also sets the `PYTHONHASHSEED` environment variable and configures PyTorch's cuDNN backend for deterministic behavior.

    Args:
        seed_value (int): The seed value to use for all random number generators.

    Returns:
        None
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
