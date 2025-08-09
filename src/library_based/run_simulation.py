import multiprocessing
import os
import pickle
import time
from datetime import datetime

import folium
import torch
from folium.plugins import MarkerCluster
from IPython.display import display, Markdown

from .agent import generate_dispatch_with_hf_agent
from .constants import *
from .environment import BikeRedistributionEnv
from .routing import (find_path_worker, init_worker, plan_optimized_journeys, schedule_journeys)
from .train import train_or_load_model
from .utils import (build_or_load_nyc_graph, calculate_thresholds, load_and_filter, load_stations, login_to_huggingface,
                    preprocess_demand_data, sanitize_for_json)
from .visualisation import (draw_route, inject_corrected_animation_js, map_stations_to_nodes, plot_stations_cluster)


def run_simulation(
        inventory_model_path: str = MODEL_PATH,
        graph_path: str = NYC_GRAPH_PATH
):
    """
    Runs the full SmartFlow bike redistribution simulation pipeline.

    Steps performed:
    1. Loads and preprocesses demand and station data.
    2. Builds or loads the NYC road network graph.
    3. Trains or loads the inventory RL agent for bike redistribution.
    4. Runs the agent to generate initial transfer plans.
    5. Plans and schedules truck journeys for redistribution.
    6. Logs pre-simulation dispatch details.
    7. Builds and animates a map of the simulation.
    8. Generates an agentic dispatch report using HuggingFace.
    9. Saves simulation results for later analysis.

    Args:
        inventory_model_path (str): Path to the RL agent model file.
        graph_path (str): Path to the NYC road network graph file.

    Returns:
        None. Saves results and visualizations to disk.
    """
    # 1. Load Data & Configure
    print("Step 1: Loading and filtering data...")
    df = load_and_filter(TARGET_DATE)
    full_trips_df = pd.read_csv(TRIPS_FILE, parse_dates=["Start Time", "Stop Time"])
    demand_data = preprocess_demand_data(full_trips_df)
    stations_meta = load_stations()
    counts_all = df[COL_START_NAME].value_counts()
    valid_stations = [s for s in counts_all.index if s in stations_meta.index]
    top = valid_stations[:TOP_N]
    vol = counts_all.loc[top]
    capacities = stations_meta["capacity"].to_dict()
    thresh = calculate_thresholds(vol, capacities, df.get(COL_PRECIP, pd.Series(dtype=float)).mean())
    coord_df = df.dropna(subset=[COL_START_LAT, COL_START_LON]).groupby(COL_START_NAME)[
        [COL_START_LAT, COL_START_LON]].mean()
    coords_top = {s: (float(coord_df.at[s, COL_START_LAT]), float(coord_df.at[s, COL_START_LON])) for s in top if
                  s in coord_df.index}
    print("✅ Data loading complete.")

    # 2. Build or Load Graph
    print("\nStep 2: Building or loading road network graph...")
    G = build_or_load_nyc_graph(coords_top, file_path=graph_path)
    station_to_node = map_stations_to_nodes(G, stations_meta.loc[top], lat_col=COL_START_LAT, lon_col=COL_START_LON)
    print("✅ Graph loaded.")

    # 3. Train or Load the Inventory RL Agent
    print("\nStep 3: Training or loading the inventory model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_envs = max(1, os.cpu_count() - 1)
    inventory_model = train_or_load_model(
        top=top, thresholds=thresh, capacities=capacities, coords_all=coords_top,
        demand_data=demand_data, n_envs=n_envs, device=device, model_path=inventory_model_path
    )

    # 4. Run Inventory Model to Get Strategic Plan
    print("\nStep 4: Running inventory model to determine initial transfers...")
    inv_env = BikeRedistributionEnv(
        stations=top, thresholds=thresh, capacities=capacities, coords=coords_top,
        demand_data=demand_data, max_steps=MAX_STEPS, gamma=GAMMA
    )
    obs, _ = inv_env.reset()
    initial_counts = {s: int(obs[i]) for i, s in enumerate(top)}

    transfers = {}
    done = False
    step_counter = 0
    while not done:
        current_hour = obs[-1]
        action, _ = inventory_model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = inv_env.step(int(action))

        if reward >= 0:
            src_idx, tgt_idx = inv_env.actions[info.get("exec_action", int(action))]
            if (src_idx, tgt_idx) not in transfers:
                transfers[(src_idx, tgt_idx)] = {'count': 0, 'first_hour': current_hour}
            transfers[(src_idx, tgt_idx)]['count'] += 1
        step_counter += 1
        print(f"\r   - Simulating step: {step_counter}/{MAX_STEPS}", end="")
        done = term or trunc
    print()
    print(f"✅ Initial transfers planned: {len(transfers)} unique routes suggested.")

    # 5. Plan and Schedule Journeys
    optimized_journeys, live_counts = plan_optimized_journeys(initial_counts, thresh, top)
    optimized_journeys = schedule_journeys(optimized_journeys, transfers, top)

    # 6. Pre-simulation Dispatch Log
    print("\n--- PRE-SIMULATION DISPATCH LOG ---")
    for journey in optimized_journeys:
        truck_id = journey['truck_id']
        print(f"\n--- Dispatching Truck_{truck_id} ---")
        for leg in journey['legs']:
            dispatch_time = leg.get('dispatch_time', 'Not Scheduled')
            depart_msg = f"▶️ {dispatch_time} - Truck_{truck_id}: Departing {leg['src']} ({leg['move']} bikes)"
            arrive_msg = f"✅ Truck_{truck_id}: Arrived at {leg['tgt']}"
            print(depart_msg)
            print(arrive_msg)

    # 7. Build and Animate Map
    print("\nStep 7: Building and rendering the final map...")
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    cluster = MarkerCluster().add_to(m)
    plot_stations_cluster(cluster, coords_top, initial_counts, thresh)
    truck_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'pink', 'black']

    print("   - Calculating all route paths in parallel...")
    pathfinding_tasks = [(leg["src"], leg["tgt"]) for journey in optimized_journeys for leg in journey["legs"]]
    with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker, initargs=(G, station_to_node)) as pool:
        path_results = pool.map(find_path_worker, pathfinding_tasks)
    path_dict = {(src, tgt): nodes for src, tgt, nodes in path_results if nodes}
    print(f"   - ✅ {len(path_dict)} paths found successfully.")

    animation_data = []
    final_counts_for_anim = initial_counts.copy()
    for journey in optimized_journeys:
        truck_id = journey["truck_id"]
        color = truck_colors[truck_id % len(truck_colors)]
        for leg in journey["legs"]:
            src, tgt, move = leg["src"], leg["tgt"], leg["move"]
            path_nodes = path_dict.get((src, tgt))
            if not path_nodes: continue

            count_src_before_leg = final_counts_for_anim[src]
            final_counts_for_anim[src] -= move
            final_counts_for_anim[tgt] += move

            animation_data.append({
                "truck_id": truck_id, "src_name": src, "tgt_name": tgt,
                "path_nodes": path_nodes, "bikes_on_truck_start": move,
                "final_count_src": count_src_before_leg - move,
                "final_count_tgt": final_counts_for_anim[tgt],
            })
            draw_route(m, G, coords_top[src][0], coords_top[src][1], coords_top[tgt][0], coords_top[tgt][1],
                       color=color)

    inject_animation_js(m, animation_data, G)
    out_path = "SmartFlow_Final_Simulation.html"
    m.save(out_path)
    print(f"🗺️ Map saved to → {out_path}")
    display(m)

    # 8. Generate Agentic Report
    print("\n\nStep 8: Generating Agentic Dispatch and Reporting...")
    login_to_huggingface()
    start_time = time.time()
    print(f"   - Agentic layer started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sanitized_journeys = sanitize_for_json(optimized_journeys)
    with open("sanitized_journeys.json", "w") as jf:
        json.dump(sanitized_journeys, jf, indent=2)
    dispatch_report = generate_dispatch_with_hf_agent(sanitized_journeys)
    end_time = time.time()
    duration = end_time - start_time
    print(f"   - Agentic layer finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   - Total time taken for report generation: {duration:.2f} seconds")
    print("✅ Report generation complete.")
    display(Markdown(dispatch_report))

    # 9. Save Results for Analysis
    print("\nStep 9: Saving simulation results for later analysis...")
    results_to_save = {
        "initial_counts": initial_counts, "live_counts": live_counts, "thresh": thresh,
        "optimized_journeys": optimized_journeys, "transfers": transfers, "graph": G,
        "station_to_node_map": station_to_node, "station_names": top
    }
    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(results_to_save, f)
    print("✅ Results saved to simulation_results.pkl")


if __name__ == "__main__":
    run_simulation()
