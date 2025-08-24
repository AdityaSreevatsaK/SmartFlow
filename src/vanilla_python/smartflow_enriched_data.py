import json
import logging
import random
import time
from sqlite3 import Date
from typing import Dict, Hashable, List, Tuple

import folium
import requests
from branca.element import Element
from folium.plugins import MarkerCluster

from vanilla_python.constants import *

# ── Logging setup ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Phase 1a: Data loading & thresholds ────────────────────────────────────────
def load_and_filter(date: Date) -> pd.DataFrame:
    df = pd.read_csv(
        TRIPS_FILE,
        parse_dates=["Start Time", "Stop Time"],
        low_memory=False,
    )
    # drop incomplete
    df = df.dropna(subset=[COL_START_NAME, COL_END_NAME])

    # strip any leading/trailing spaces
    df[COL_START_NAME] = df[COL_START_NAME].str.strip()
    df[COL_END_NAME] = df[COL_END_NAME].str.strip()

    df = df[df["Start Time"].dt.date == date]
    logger.info("Loaded & filtered %d enriched trips on %s", len(df), date)
    return df


def load_stations() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_FILE)

    # strip whitespace here too
    stations["Station Name"] = stations["Station Name"].str.strip()

    stations = stations.rename(
        columns={
            "Station Name": COL_START_NAME,
            "Latitude": COL_START_LAT,
            "Longitude": COL_START_LON,
        }
    ).set_index(COL_START_NAME)
    return stations


def get_top_stations(df: pd.DataFrame) -> List[str]:
    """
    Return the TOP_N most frequent departure stations.
    """
    top = df[COL_START_NAME].value_counts().index.tolist()[:TOP_N]
    logger.info("Top %d stations: %s", TOP_N, top)
    return top


def calculate_thresholds(
        counts: pd.Series,
        capacities: Dict[Hashable, int],
        avg_precip: float = 0.0
) -> Dict[Hashable, int]:
    """
    Heuristic threshold = max(count/8, 3), scaled down for rain (≤20%),
    and never above station capacity.
    """
    rain_factor = 1.0 - min(avg_precip / 10.0, 0.2)
    thresholds: Dict[Hashable, int] = {}
    for station, count in counts.items():
        base = max(int(count / 8 * rain_factor), 3)
        cap = capacities.get(station, base)
        thresholds[station] = min(base, cap)
    logger.info("Calculated thresholds (sample): %s", dict(list(thresholds.items())[:5]))
    return thresholds


# ── Phase 1b: Simulation & Reward ───────────────────────────────────────────────

def simulate_bike_counts(
        thresholds: Dict[Hashable, int],
        capacities: Dict[Hashable, int]
) -> Dict[Hashable, int]:
    """
    Random initial counts ∈ [0, 2*threshold], capped by capacity.
    """
    counts = {
        st: random.randint(0, min(thresholds[st] * 2, capacities.get(st, thresholds[st] * 2)))
        for st in thresholds
    }
    return counts


def compute_reward(
        source: str,
        target: str,
        bike_counts: Dict[Hashable, int],
        thresholds: Dict[Hashable, int]
) -> float:
    """
    Shaped reward combining local improvement, perfect‐balance bonus,
    network‐level shaping, and invalid‐move penalty.
    """
    # invalid if no surplus at source or no deficit at target
    if bike_counts[source] <= thresholds[source] or bike_counts[target] >= thresholds[target]:
        return -1.0

    # local improvement
    before_loc = abs(bike_counts[target] - thresholds[target])
    after_loc = abs((bike_counts[target] + 1) - thresholds[target])
    local_imp = before_loc - after_loc
    bonus = 1.0 if after_loc == 0 else 0.0

    # potential‐based shaping
    phi_before = -sum(abs(bike_counts[s] - thresholds[s]) for s in thresholds)
    bike_counts[source] -= 1
    bike_counts[target] += 1
    phi_after = -sum(abs(bike_counts[s] - thresholds[s]) for s in thresholds)
    # restore
    bike_counts[source] += 1
    bike_counts[target] -= 1

    return local_imp + bonus + (GAMMA * phi_after - phi_before)


# ── Phase 1c: Q-Learning ────────────────────────────────────────────────────────

def train_q_tables(
        stations: List[str],
        thresholds: Dict[Hashable, int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    q_source = pd.DataFrame(0.0, index=stations, columns=stations)
    q_target = pd.DataFrame(0.0, index=stations, columns=stations)
    metrics = []
    epsilon = EPSILON_START
    start_time = time.perf_counter()

    for ep in range(1, EPISODES + 1):
        bike_counts = simulate_bike_counts(thresholds, {s: thresholds[s] * 2 for s in thresholds})
        total_reward = 0.0
        steps = 0

        for _ in range(MAX_STEPS):
            feasible = [
                (i, j)
                for i in stations for j in stations
                if i != j
                   and bike_counts[i] > thresholds[i]
                   and bike_counts[j] < thresholds[j]
            ]
            if not feasible:
                break

            if random.random() < epsilon:
                source, target = random.choice(feasible)
            else:
                source, target = max(feasible, key=lambda p: q_source.at[p[0], p[1]])

            r = compute_reward(source, target, bike_counts, thresholds)
            total_reward += r
            bike_counts[source] -= 1
            bike_counts[target] += 1
            steps += 1

            # Q-source update
            old_q = q_source.at[source, target]
            best_next = q_target.loc[target].max()
            q_source.at[source, target] = (1 - ALPHA) * old_q + ALPHA * (r + GAMMA * best_next)

            # Q-target update
            old_qt = q_target.at[target, source]
            best_next_t = q_source.loc[source].max()
            q_target.at[target, source] = (1 - ALPHA) * old_qt + ALPHA * (r + GAMMA * best_next_t)

        avg_step = total_reward / steps if steps else 0.0
        metrics.append((ep, total_reward, steps, avg_step))

        if epsilon > EPSILON_END:
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if ep % LOG_INTERVAL == 0:
            recent = [m[3] for m in metrics[-LOG_INTERVAL:]]
            logger.info(
                "Episode %d/%d — avg step reward: %.4f; ε: %.4f",
                ep, EPISODES, sum(recent) / len(recent), epsilon
            )

    elapsed = time.perf_counter() - start_time
    df_metrics = pd.DataFrame(metrics, columns=["episode", "total_reward", "steps", "avg_step_reward"])
    df_metrics.to_csv(TRAINING_REWARDS, index=False)
    logger.info(
        "Training complete in %.1f s; overall avg step reward: %.4f",
        elapsed, df_metrics["avg_step_reward"].mean()
    )
    return q_source, q_target, df_metrics, elapsed


def summarise_q(q: pd.DataFrame, name: str):
    nz = (q.values != 0).sum()
    logger.info(
        "%s Q-table: min=%.2f, max=%.2f, mean=%.2f, nonzero=%d",
        name, q.values.min(), q.values.max(), q.values.mean(), nz
    )


def save_q_tables(q_source: pd.DataFrame, q_target: pd.DataFrame):
    q_source.to_csv(Q_SOURCE_FILE)
    q_target.to_csv(Q_TARGET_FILE)
    logger.info("Saved Q-tables.")


def load_q_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    q_source = pd.read_csv(Q_SOURCE_FILE, index_col=0)
    q_target = pd.read_csv(Q_TARGET_FILE, index_col=0)
    logger.info("Loaded Q-tables.")
    return q_source, q_target


# ── Phase 2: Route computation & visualisation ─────────────────────────────────

def find_dual_policy_routes_all(
        stations: List[str],
        coordinates: Dict[str, Dict[str, float]],
        q_source: pd.DataFrame,
        q_target: pd.DataFrame,
        bike_counts: Dict[Hashable, int],
        thresholds: Dict[Hashable, int]
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], int]]:
    routes, transfers = [], {}
    for src in stations:
        if src not in coordinates: continue
        for tgt, qval in q_source.loc[src].items():
            if qval <= 0 or tgt == src or tgt not in coordinates:
                continue
            if bike_counts[src] <= thresholds[src] or bike_counts[tgt] >= thresholds[tgt]:
                continue
            if q_target.loc[tgt, src] <= 0:
                continue

            need = thresholds[tgt] - bike_counts[tgt]
            avail = bike_counts[src] - thresholds[src]
            mv = min(max(need, 1), avail)

            transfers[(src, tgt)] = mv
            bike_counts[src] -= mv
            bike_counts[tgt] += mv
            routes.append((src, tgt))

    logger.info("Found %d routes.", len(routes))
    return routes, transfers


def plot_stations_cluster(
        cluster: MarkerCluster,
        coordinates: Dict[str, Dict[str, float]],
        bike_counts: Dict[Hashable, int],
        thresholds: Dict[Hashable, int]
):
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
            location=[loc[COL_START_LAT], loc[COL_START_LON]],
            icon=folium.DivIcon(html=html)
        ).add_to(cluster)


def inject_animation_js(
        m: folium.Map,
        coordinates: Dict[str, Dict[str, float]],
        routes: List[Tuple[str, str]],
        transfers: Dict[Tuple[str, str], int],
        bike_counts: Dict[Hashable, int]
):
    data = []
    for i, (source, target) in enumerate(routes, 1):
        c1 = coordinates[source]
        c2 = coordinates[target]
        url = OSRM_URL.format(
            from_lon=c1[COL_START_LON],
            from_lat=c1[COL_START_LAT],
            to_lon=c2[COL_START_LON],
            to_lat=c2[COL_START_LAT]
        )
        try:
            r = requests.get(url)
            r.raise_for_status()
            path = [[lat, lon] for lon, lat in r.json()["routes"][0]["geometry"]["coordinates"]]
            folium.PolyLine(path, color="red", weight=3).add_to(m)
            mv = transfers[(source, target)]
            data.append({
                "id": f"Truck_{i}",
                "path": path,
                "from": source.replace(" ", "_"),
                "to": target.replace(" ", "_"),
                "cs": bike_counts[source],
                "ct": bike_counts[target],
                "mv": mv
            })
        except Exception as e:
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
    # ── Phase 1: Train ─────────────────────────────────────────────────────────
    trips_data = load_and_filter(TARGET_DATE)
    stations_meta = load_stations()

    # 1) Get full ranking of stations by trip count
    counts_all = trips_data[COL_START_NAME].value_counts()

    # 2) Build a TOP_N list of VALID stations, skipping any that lack metadata
    top_stations = []
    for st in counts_all.index:
        if st in stations_meta.index:
            top_stations.append(st)
        else:
            logger.warning("No metadata for station %s – skipping", st)
        if len(top_stations) == TOP_N:
            break

    # 3) Now top_stations is guaranteed length TOP_N and all have metadata
    logger.info("Final top %d stations: %s", TOP_N, top_stations)

    # 4) Compute trip volumes just on those stations
    trip_vol = counts_all.loc[top_stations]

    avg_precip = trips_data.get("precip", pd.Series()).mean()
    capacities = stations_meta["capacity"].to_dict()
    thresholds = calculate_thresholds(trip_vol, capacities, avg_precip)

    q_source, q_target, df_metrics, elapsed = train_q_tables(top_stations, thresholds)
    summarise_q(q_source, name="Source")
    summarise_q(q_target, name="Target")
    save_q_tables(q_source, q_target)

    # ── Phase 2: Visualise ───────────────────────────────────────────────────────
    coordinates = {
        st: {
            COL_START_LAT: stations_meta.at[st, COL_START_LAT],
            COL_START_LON: stations_meta.at[st, COL_START_LON]
        }
        for st in top_stations
    }

    bike_counts = simulate_bike_counts(thresholds, capacities)
    routes, transfers = find_dual_policy_routes_all(
        top_stations, coordinates, q_source, q_target, bike_counts, thresholds
    )

    folium_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    marker_cluster = MarkerCluster().add_to(folium_map)
    plot_stations_cluster(marker_cluster, coordinates, bike_counts, thresholds)
    inject_animation_js(folium_map, coordinates, routes, transfers, bike_counts)
    folium_map.save(MAP_OUTPUT)
    logger.info("Map saved to %s", MAP_OUTPUT)


if __name__ == "__main__":
    main()
