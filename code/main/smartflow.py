import json
import logging
import random
import time
from sqlite3 import Date
from typing import Dict, Hashable, List, Tuple

import folium
import pandas as pd
import requests
from branca.element import Element
from folium.plugins import MarkerCluster

# ── Config ───────────────────────────────────────────────────────────────────
DATA_FILE = "../../data/raw/New York CitiBike - 2015-2017.csv"
Q_SOURCE_FILE = "../../data/processed/Q Table - Source.csv"
Q_TARGET_FILE = "../../data/processed/Q Table - Target.csv"
TRAINING_REWARDS = "../../results/rewards/SmartFlow - Rewards.csv"
MAP_OUTPUT = "../../results/simulation/SmartFlow - Simulation.html"

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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Phase 1: Q‑Learning ─────────────────────────────────────────────────────────
# ── Phase 1a: Data & Thresholds ─────────────────────────────────────────────────
def load_and_filter(date: Date) -> pd.DataFrame:
    """
    Loads trip data from CSV, cleans it, and filters for a specific date.

    Args:
        date: The target date to filter the trip data for.

    Returns:
        A pandas DataFrame containing trip data for the specified date.
    """
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["Start Station Name", "End Station Name"])
    df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
    df = df[df["Start Time"].dt.date == date]
    logger.info("Loaded & filtered %d trips on %s", len(df), date)
    return df


def get_top_stations(df: pd.DataFrame) -> List[str]:
    """
    Identifies the most popular starting stations based on trip volume.

    Args:
        df: The DataFrame containing trip data.

    Returns:
        A list of station names, sorted by popularity (most trips first).
    """
    top_stations = df["Start Station Name"].value_counts().index.tolist()[:TOP_N]
    logger.info("Top stations: %s", top_stations)
    return top_stations


def calculate_thresholds(counts: pd.Series) -> dict[Hashable, int]:
    """
    Calculates the bike count threshold for each station.

    The threshold is a heuristic value (trip count // 8, with a minimum of 3)
    used to determine if a station has a surplus or deficit of bikes.

    Args:
        counts: A pandas Series with station names as index and trip counts as values.

    Returns:
        A dictionary mapping each station name to its calculated threshold.
    """
    return {st: max(int(c / 8), 3) for st, c in counts.items()}


# ── Phase 1b: Simulation & Reward ───────────────────────────────────────────────
def simulate_bike_counts(thresholds: Dict[Hashable, int]) -> Dict[Hashable, int]:
    """
    Generate initial bike counts so that some stations start below,
    some at, and some above their threshold.

    Args:
        thresholds: A dict mapping station names to their threshold values.

    Returns:
        A dict mapping station names to a random initial bike count in [0, 2*threshold].
    """
    return {
        st: random.randint(0, thresholds[st] * 2)
        for st in thresholds
    }


def compute_reward(
        source: str,
        target: str,
        bike_counts: Dict[Hashable, int],
        thresholds: Dict[Hashable, int]
) -> float:
    """
    Computes a shaped reward that combines:
      1) Local improvement at the target station,
      2) A bonus for perfect balance at the target,
      3) Potential-based shaping to reduce overall network imbalance,
      4) A penalty for invalid moves.

    Args:
        source:      Name of the source station.
        target:      Name of the target station.
        bike_counts: Current bike counts at each station (modified then restored).
        thresholds:  Desired bike thresholds for each station.

    Returns:
        A float reward value.
    """
    # 1) Invalid moves penalised
    if bike_counts[source] <= thresholds[source]:
        return -1.0
    if bike_counts[target] >= thresholds[target]:
        return -1.0

    # 2) Local improvement
    before_loc = abs(bike_counts[target] - thresholds[target])
    after_loc = abs((bike_counts[target] + 1) - thresholds[target])
    local_imp = before_loc - after_loc
    bonus = 1.0 if after_loc == 0 else 0.0

    # 3) Potential-based shaping (network-level)
    phi_before = -sum(abs(bike_counts[s] - thresholds[s]) for s in thresholds)
    bike_counts[source] -= 1
    bike_counts[target] += 1
    phi_after = -sum(abs(bike_counts[s] - thresholds[s]) for s in thresholds)
    bike_counts[source] += 1
    bike_counts[target] -= 1

    return local_imp + bonus + (GAMMA * phi_after - phi_before)


# ── Phase 1c: Q‑Learning ───────────────────────────────────────────────────────
def train_q_tables(
        stations: List[str],
        thresholds: Dict[Hashable, int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Trains the source and target Q‑tables via Q‑learning, tracking per‑episode
    metrics including total reward, steps taken, and average step reward.

    Returns:
      - q_source, q_target: trained Q‑table DataFrames
      - df_metrics: DataFrame with columns [episode, total_reward, steps, avg_step_reward]
      - elapsed: training time in seconds
    """
    q_source = pd.DataFrame(0.0, index=stations, columns=stations)
    q_target = pd.DataFrame(0.0, index=stations, columns=stations)

    metrics = []  # Will hold (episode, total_reward, steps, avg_step_reward)
    epsilon = EPSILON_START
    start_time = time.perf_counter()

    for ep in range(1, EPISODES + 1):
        bike_counts = simulate_bike_counts(thresholds)
        total_reward = 0.0
        steps_executed = 0

        for _ in range(MAX_STEPS):
            # Feasible moves: surplus -> deficit
            feasible = [
                (i, j)
                for i in stations
                for j in stations
                if i != j
                   and bike_counts[i] > thresholds[i]
                   and bike_counts[j] < thresholds[j]
            ]
            if not feasible:
                break

            # ε‑greedy selection
            if random.random() < epsilon:
                source, target = random.choice(feasible)
            else:
                source, target = max(
                    feasible,
                    key=lambda p: q_source.at[p[0], p[1]]
                )

            # Apply move & compute reward
            r = compute_reward(source, target, bike_counts, thresholds)
            total_reward += r
            bike_counts[source] -= 1
            bike_counts[target] += 1
            steps_executed += 1

            # Q‑source update
            old_q = q_source.loc[source, target]
            best_next = q_target.loc[target].max()
            q_source.loc[source, target] = (1 - ALPHA) * old_q + ALPHA * (r + GAMMA * best_next)

            # Q‑target update
            old_qt = q_target.loc[target, source]
            best_next_t = q_source.loc[source].max()
            q_target.loc[target, source] = (1 - ALPHA) * old_qt + ALPHA * (r + GAMMA * best_next_t)

        # Record metrics for this episode
        avg_step_reward = total_reward / steps_executed if steps_executed else 0.0
        metrics.append((ep, total_reward, steps_executed, avg_step_reward))

        # Decay epsilon
        if epsilon > EPSILON_END:
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Periodic logging of avg step reward
        if ep % LOG_INTERVAL == 0:
            recent = [m[3] for m in metrics[-LOG_INTERVAL:]]
            logger.info(
                "Episode %d/%d — avg step reward: %.4f; ε: %.4f",
                ep, EPISODES, sum(recent) / len(recent), epsilon
            )

    elapsed = time.perf_counter() - start_time

    # Build and save DataFrame.
    df_metrics = pd.DataFrame(
        metrics,
        columns=["episode", "total_reward", "steps", "avg_step_reward"]
    )
    df_metrics.to_csv(TRAINING_REWARDS, index=False)
    logger.info(
        "Training complete in %.1f s; overall avg step reward: %.4f",
        elapsed, df_metrics["avg_step_reward"].mean()
    )

    return q_source, q_target, df_metrics, elapsed


def summarise_q(q: pd.DataFrame, name: str):
    """
    Logs summary statistics for a Q‑table DataFrame, including the minimum,
    maximum, mean Q‑value and the count of non‑zero entries.

    Args:
        q:     A pandas DataFrame representing the Q‑table, where both the
               index and columns correspond to station names.
        name:  A string label for the table (e.g. "Source" or "Target"), used
               to prefix the log message.

    Returns:
        None.  Emits an INFO‑level log containing:
          - Minimum Q‑value across all entries.
          - Maximum Q‑value across all entries.
          - Mean Q‑value.
          - Number of non‑zero Q‑values.
    """
    nz = (q.values != 0).sum()
    logger.info(
        "%s Q‑table: min=%.2f, max=%.2f, mean=%.2f, nonzero=%d",
        name, q.values.min(), q.values.max(),
        q.values.mean(), nz
    )


def save_q_tables(
        q_source: pd.DataFrame,
        q_target: pd.DataFrame
) -> None:
    """
    Persists the trained source and target Q‑tables to CSV files.

    Args:
        q_source: A pandas DataFrame representing the source Q‑table,
                  with source stations as the index and target stations as columns.
        q_target: A pandas DataFrame representing the target Q‑table,
                  with target stations as the index and source stations as columns.

    Returns:
        None. Writes `q_source` to `Q_SOURCE_FILE` and `q_target` to `Q_TARGET_FILE`,
        logging an INFO‑level message upon success.
    """
    q_source.to_csv(Q_SOURCE_FILE)
    q_target.to_csv(Q_TARGET_FILE)
    logger.info("Saved Q‑tables.")


def load_q_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the source and target Q‑tables from CSV files into DataFrames.

    Returns:
        A tuple `(q_source, q_target)` where:
          - q_source: DataFrame loaded from `Q_SOURCE_FILE`, indexed by source station.
          - q_target: DataFrame loaded from `Q_TARGET_FILE`, indexed by target station.

    Notes:
        The function logs an INFO‑level message once both tables are successfully loaded.
    """
    q_source = pd.read_csv(Q_SOURCE_FILE, index_col=0)
    q_target = pd.read_csv(Q_TARGET_FILE, index_col=0)
    logger.info("Loaded Q‑tables.")
    return q_source, q_target


# ── Phase 2: Visualization ───────────────────────────────────────────────────────
def get_station_coordinates(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Extracts the geographic coordinates for each station from the trip DataFrame.

    Groups the DataFrame by 'Start Station Name' and takes the first
    observed latitude and longitude for each station, under the assumption
    that all records for a given station share the same coordinates.

    Args:
        df: A pandas DataFrame containing at least the columns
            'Start Station Name', 'Start Station Latitude',
            and 'Start Station Longitude'.

    Returns:
        A dictionary where each key is a station name (string) and each value
        is a sub-dictionary with:
            {
                'Start Station Latitude': <float>,
                'Start Station Longitude': <float>
            }
    """
    grp = df.groupby("Start Station Name")[
        ["Start Station Latitude", "Start Station Longitude"]
    ].first()
    return grp.to_dict("index")


def find_dual_policy_routes_all(stations, coordinates, q_source, q_target, bike_counts, thresholds):
    routes, transfers = [], {}
    for src in stations:
        if src not in coordinates: continue
        # look at _every_ target with positive Q‑value
        for tgt, qval in q_source.loc[src].items():
            if qval <= 0 or tgt == src or tgt not in coordinates:
                continue

            # surplus / deficit check
            if bike_counts[src] <= thresholds[src] or bike_counts[tgt] >= thresholds[tgt]:
                continue

            # reciprocal check
            if q_target.loc[tgt, src] <= 0:
                continue

            # compute how many to move
            need = thresholds[tgt] - bike_counts[tgt]
            avail = bike_counts[src] - thresholds[src]
            mv = min(max(need, 1), avail)

            key = (src, tgt)
            transfers[key] = mv
            bike_counts[src] -= mv
            bike_counts[tgt] += mv
            routes.append((src, tgt))

    logger.info("Found %d routes.", len(routes))
    return routes, transfers


def plot_stations_cluster(cluster, coordinates, bike_counts, thresholds):
    """
    Adds coloured station markers to a Folium MarkerCluster based on current
    bike counts versus thresholds.

    For each station in `coordinates`:
      1. Retrieves its current count from `bike_counts` (defaults to 0).
      2. Chooses marker colour:
         - "green" if count ≥ threshold (surplus)
         - "orange" if count < threshold (deficit)
      3. Builds a small HTML snippet showing the station identifier and count.
      4. Places a DivIcon marker at the station’s latitude/longitude on `cluster`.

    Args:
        cluster:      A Folium MarkerCluster instance to which markers are added.
        coordinates:  Mapping from station name to a dict containing
                      `"Start Station Latitude"` and `"Start Station Longitude"`.
        bike_counts:  Mapping from station name to its simulated/final bike count.
        thresholds:   Mapping from station name to its target threshold count.

    Returns:
        None.  Markers are added directly to the provided `cluster`.
    """
    for name, loc in coordinates.items():
        cnt = bike_counts.get(name, 0)
        col = "green" if cnt >= thresholds[name] else "orange"
        safe = name.replace(" ", "_")
        html = (
            f"<div style='white-space:nowrap;text-align:center;'>"
            f"<span id='count_{safe}' style='background:{col};"
            f"color:white;padding:4px 6px;border-radius:4px;font-size:11px;'>"
            f"{safe} 🚲 {cnt}</span>"
            f"<img src='{BIKE_ICON_URL}' width='24' height='24'/>"
            "</div>"
        )
        folium.Marker(
            location=[loc["Start Station Latitude"], loc["Start Station Longitude"]],
            icon=folium.DivIcon(html=html)
        ).add_to(cluster)


def inject_animation_js(m, coordinates, routes, transfers, bike_counts):
    """
    Injects a JavaScript animation into a Folium map to visualise truck
    movements and live station updates for each planned bike transfer.

    This function:
      1. For each (source, target) pair in `routes`:
         - Looks up the start/end coordinates from `coordinates`.
         - Queries the OSRM API to retrieve the driving path geometry.
         - Draws a red polyline for that route on the map `m`.
         - Collects the path, station IDs, pre-/post-bike counts and number
           of bikes moved into a JSON-serialisable list.
      2. Constructs an HTML <script> that, on map load:
         - Creates a fixed-position “banner” div for status messages.
         - Defines `animateTruck()`, which moves a truck icon along the path,
           updates the banner, and updates the station count markers on arrival.
         - Iterates through each route dataset and calls `animateTruck()`.
      3. Inserts the script into `m` using `branca.element.Element`.

    Args:
        m:           A Folium Map instance to which the animation will be added.
        coordinates: A mapping from station name to a dict with keys
                     `"Start Station Latitude"` and `"Start Station Longitude"`.
        routes:      A list of `(source_name, target_name)` tuples defining
                     each bike-transfer route.
        transfers:   A dict mapping the concatenated `source+target` key to
                     the integer number of bikes to move along that route.
        bike_counts: A dict of final bike counts at each station (after
                     transfers), used to update marker labels.

    Returns:
        None.  The map `m` is modified in-place.

    Notes:
        - Any exceptions from the OSRM request (e.g. RequestException,
          KeyError, IndexError, TypeError) are caught and logged; they do
          not interrupt insertion of the animation script.
    """
    data = []
    for i, (source, target) in enumerate(routes, 1):
        c1, c2 = coordinates[source], coordinates[target]
        url = OSRM_URL.format(
            from_lon=c1["Start Station Longitude"],
            from_lat=c1["Start Station Latitude"],
            to_lon=c2["Start Station Longitude"],
            to_lat=c2["Start Station Latitude"]
        )
        try:
            r = requests.get(url)
            r.raise_for_status()
            path = [[lat, lon] for lon, lat in r.json()["routes"][0]["geometry"]["coordinates"]]
            folium.PolyLine(path, color="red", weight=3).add_to(m)
            key = (source, target)
            data.append({
                "id": f"Truck_{i}", "path": path,
                "from": source.replace(" ", "_"), "to": target.replace(" ", "_"),
                "cs": bike_counts[source], "ct": bike_counts[target], "mv": transfers[key]
            })
        except (requests.RequestException, KeyError, IndexError, TypeError) as e:
            logger.exception("OSRM fail %s→%s: %s", source, target, e)

    js_data = json.dumps(data)
    map_var = m.get_name()
    js = f"""
    <script>
    document.addEventListener('DOMContentLoaded',function(){{
      var map=window["{map_var}"];
      var banner=document.createElement('div');
      Object.assign(banner.style,{{
        position:'fixed',bottom:'10px',right:'10px',
        padding:'8px',background:'rgba(0,0,0,0.7)',
        color:'white',fontSize:'12px',whiteSpace:'pre',zIndex:10000
      }});
      banner.innerText="Truck animation status.\\n";
      document.body.appendChild(banner);

      function animateTruck(id,path,dur,fromSt,toSt,cSrc,cTgt,moved){{
        banner.innerText+=id+" - Departed from source station.\\n";
        var iconHtml=
          "<div style='text-align:center;white-space:nowrap;display:inline-block;'>"+
          "<div style='background:black;color:white;padding:4px;border-radius:4px;font-size:11px;'>"+
          id+" 🚲 "+moved+"</div>"+
          "<img src='{TRUCK_ICON_URL}' width='30' height='30'/></div>";
        var marker=L.marker(path[0],{{icon:L.divIcon({{html:iconHtml}})}}).addTo(map);
        var i=0,steps=path.length,delay=(dur*1000)/steps;
        (function mv(){{
          if(i<steps){{marker.setLatLng(path[i++]);setTimeout(mv,delay);}}
          else {{
            banner.innerText+=id+" - Arrived at target station.\\n";
            var t=document.getElementById('count_'+toSt),
                s=document.getElementById('count_'+fromSt);
            if(t){{t.style.background='green';t.innerText=toSt+" 🚲 "+cTgt;}}
            if(s){{s.style.background='green';s.innerText=fromSt+" 🚲 "+cSrc;}}
          }}
        }})();
      }}

      var routes={js_data};
      routes.forEach(r=>animateTruck(r.id,r.path,10,r.from,r.to,r.cs,r.ct,r.mv));
    }});
    </script>
    """
    m.get_root().html.add_child(Element(js))


def main():
    """
    Orchestrates the end‑to‑end SmartFlow workflow in two phases:

      Phase 1: Model training
        1. Load and filter CitiBike trip data for TARGET_DATE.
        2. Select the top TOP_N departure stations.
        3. Compute bike‑inventory thresholds for each station.
        4. Train dual Q‑learning tables with per‑step metrics.
        5. Log Q‑table summaries and save them to disk.

      Phase 2: Route computation and visualisation
        1. Reload the trained Q‑tables.
        2. Extract station coordinates from the filtered DataFrame.
        3. Simulate initial bike counts.
        4. Identify optimal redistribution routes via dual‑policy matching.
        5. Create a Folium map with clustered station markers.
        6. Inject JavaScript to animate truck movements and update counts.
        7. Save the interactive HTML map to MAP_OUTPUT.

    Returns:
        None.  Side effects include logging progress, writing Q‑tables and
        metrics CSV, and exporting the final HTML map.
    """
    # Phase 1: Train
    trips_data = load_and_filter(TARGET_DATE)
    top_stations = get_top_stations(trips_data)
    trip_vol = trips_data[trips_data["Start Station Name"].isin(top_stations)]["Start Station Name"].value_counts()
    thresholds = calculate_thresholds(trip_vol)

    q_source, q_target, _, _ = train_q_tables(top_stations, thresholds)
    summarise_q(q_source, name="Source")
    summarise_q(q_target, name="Target")
    save_q_tables(q_source, q_target)

    # Phase 2: Visualise
    coordinate_data = trips_data.dropna(subset=["Start Station Latitude", "Start Station Longitude"])
    q_source, q_target = load_q_tables()
    coordinates = get_station_coordinates(coordinate_data)
    coordinates = {s: coordinates[s] for s in top_stations if s in coordinates}

    bike_counts = simulate_bike_counts(thresholds)
    routes, transfers = find_dual_policy_routes_all(top_stations, coordinates, q_source, q_target, bike_counts, thresholds)

    # Start the clock for map rendering
    start_map = time.perf_counter()
    folium_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    marker_cluster = MarkerCluster().add_to(folium_map)
    plot_stations_cluster(marker_cluster, coordinates, bike_counts, thresholds)
    inject_animation_js(folium_map, coordinates, routes, transfers, bike_counts)
    folium_map.save(MAP_OUTPUT)
    logger.info("Map saved to %s", MAP_OUTPUT)

    # Stop the clock and log it
    map_time = time.perf_counter() - start_map
    logger.info("Map rendering + save took %.2f s", map_time)


if __name__ == "__main__":
    main()
