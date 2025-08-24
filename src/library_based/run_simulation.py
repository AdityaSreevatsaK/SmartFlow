import json
import multiprocessing
import os
import pickle
import time
from datetime import datetime

import folium
import torch
from folium.plugins import MarkerCluster

from .agent import generate_dispatch_with_hf_agent
from .constants import *
from .environment import BikeRedistributionEnv
from .routing import (find_path_worker, init_worker, plan_optimised_journeys, schedule_journeys)
from .train import train_or_load_model
from .utils import (build_or_load_nyc_graph, calculate_thresholds, load_and_filter, load_stations, login_to_huggingface,
                    preprocess_demand_data, sanitize_for_json, set_seed)
from .visualisation import (draw_route, inject_animation_js, map_stations_to_nodes, plot_stations_cluster)


def run_simulation(
        inventory_model_path: str = MODEL_PATH,
        graph_path: str = NYC_GRAPH_PATH,
        seed_value: int = 0
):
    """
    Runs the full bike redistribution simulation pipeline.

    Args:
        inventory_model_path (str): Path to the RL inventory model file.
        graph_path (str): Path to the road network graph file.
        seed_value (int): Random seed for reproducibility.

    Workflow:
        1. Loads and preprocesses demand and station data.
        2. Builds or loads the road network graph.
        3. Trains or loads the RL agent for inventory management.
        4. Runs the agent to generate initial bike transfer plans.
        5. Plans and schedules truck journeys for redistribution.
        6. Logs pre-simulation dispatch details.
        7. Builds and animates a folium map of the simulation.
        8. Generates an agentic dispatch report using HuggingFace.
        9. Saves simulation results for later analysis.

    Outputs:
        - Saves an HTML map visualisation.
        - Saves a JSON file of sanitised journeys.
        - Saves a pickle file with simulation results.
        - Displays dispatch logs and agentic report.
    """
    # 1. Load Data & Configure
    set_seed(seed_value)
    print(f"Step 1: Loading and filtering data for Seed {seed_value}...")
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
    coordinates_top = {s: (float(coord_df.at[s, COL_START_LAT]), float(coord_df.at[s, COL_START_LON])) for s in top if
                       s in coord_df.index}
    print("✅ Data loading complete.")

    # 2. Build or Load Graph
    print("\nStep 2: Building or loading road network graph...")
    G = build_or_load_nyc_graph(coordinates_top, file_path=graph_path)
    station_to_node = map_stations_to_nodes(G, stations_meta.loc[top], lat_col=COL_START_LAT, lon_col=COL_START_LON)
    print("✅ Graph loaded.")

    # 3. Train or Load the Inventory RL Agent
    print("\nStep 3: Training or loading the inventory model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_envs = max(1, os.cpu_count() - 1)
    inventory_model = train_or_load_model(
        top=top, thresholds=thresh, capacities=capacities, coordinates_all=coordinates_top,
        demand_data=demand_data, n_envs=n_envs, device=device,
        model_path=inventory_model_path,
        seed_value=seed_value
    )

    # 4. Run Inventory Model to Get Strategic Plan
    print("\nStep 4: Running inventory model to determine initial transfers...")
    inv_env = BikeRedistributionEnv(
        stations=top, thresholds=thresh, capacities=capacities, coordinates=coordinates_top,
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
            source_idx, target_idx = inv_env.actions[info.get("exec_action", int(action))]
            if (source_idx, target_idx) not in transfers:
                transfers[(source_idx, target_idx)] = {'count': 0, 'first_hour': current_hour}
            transfers[(source_idx, target_idx)]['count'] += 1
        step_counter += 1
        print(f"\r   - Simulating step: {step_counter}/{MAX_STEPS}", end="")
        done = term or trunc
    print()
    print(f"✅ Initial transfers planned: {len(transfers)} unique routes suggested.")

    # 5. Plan and Schedule Journeys
    optimised_journeys, live_counts = plan_optimised_journeys(initial_counts, thresh, top)
    optimised_journeys = schedule_journeys(optimised_journeys, transfers, top)

    # 6. Pre-simulation Dispatch Log
    print("\n--- PRE-SIMULATION DISPATCH LOG ---")
    for journey in optimised_journeys:
        truck_id = journey['truck_id']
        print(f"\n--- Dispatching Truck_{truck_id} ---")
        for leg in journey['legs']:
            dispatch_time = leg.get('dispatch_time', 'Not Scheduled')
            depart_msg = f"▶️ {dispatch_time} - Truck_{truck_id}: Departing {leg['source']} ({leg['move']} bikes)"
            arrive_msg = f"✅ Truck_{truck_id}: Arrived at {leg['target']}"
            print(depart_msg)
            print(arrive_msg)

    # 7. Build and Animate Map
    print("\nStep 7: Building and rendering the final map...")
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    cluster = MarkerCluster().add_to(m)
    plot_stations_cluster(cluster, coordinates_top, initial_counts, thresh)
    truck_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'pink', 'black']

    print("   - Calculating all route paths in parallel...")
    pathfinding_tasks = [(leg["source"], leg["target"]) for journey in optimised_journeys for leg in journey["legs"]]
    with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker, initargs=(G, station_to_node)) as pool:
        path_results = pool.map(find_path_worker, pathfinding_tasks)
    path_dict = {(source, target): nodes for source, target, nodes in path_results if nodes}
    print(f"   - ✅ {len(path_dict)} paths found successfully.")

    animation_data = []
    final_counts_for_anim = initial_counts.copy()
    for journey in optimised_journeys:
        truck_id = journey["truck_id"]
        color = truck_colors[truck_id % len(truck_colors)]
        for leg in journey["legs"]:
            source, target, move = leg["source"], leg["target"], leg["move"]
            path_nodes = path_dict.get((source, target))
            if not path_nodes: continue

            count_source_before_leg = final_counts_for_anim[source]
            final_counts_for_anim[source] -= move
            final_counts_for_anim[target] += move

            animation_data.append({
                "truck_id": truck_id, "source_name": source, "target_name": target,
                "path_nodes": path_nodes, "bikes_on_truck_start": move,
                "final_count_source": count_source_before_leg - move,
                "final_count_target": final_counts_for_anim[target],
            })
            draw_route(m, G, coordinates_top[source][0], coordinates_top[source][1], coordinates_top[target][0],
                       coordinates_top[target][1],
                       color=color)

    inject_animation_js(m, animation_data, G)
    out_path = RESULTS_FOLDER / f"simulation/SmartFlow_Final_Simulation_seed_{seed_value}.html"
    m.save(out_path)
    print(f"🗺️ Map saved to → {out_path}")

    # 8. Generate Agentic Report
    print("\n\nStep 8: Generating Agentic Dispatch and Reporting...")
    login_to_huggingface()
    start_time = time.time()
    print(f"   - Agentic layer started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sanitised_journeys = sanitize_for_json(optimised_journeys)
    with open(f"sanitised_journeys_seed_{seed_value}.json", "w") as jf:
        json.dump(sanitised_journeys, jf, indent=2)
    dispatch_report = generate_dispatch_with_hf_agent(sanitised_journeys)
    end_time = time.time()
    duration = end_time - start_time
    print(f"   - Agentic layer finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   - Total time taken for report generation: {duration:.2f} seconds")
    print("✅ Report generation complete.")
    print(dispatch_report)

    # 9. Save Results for Analysis
    print("\nStep 9: Saving simulation results for later analysis...")
    results_to_save = {
        "initial_counts": initial_counts, "live_counts": live_counts, "thresh": thresh,
        "optimised_journeys": optimised_journeys, "transfers": transfers, "graph": G,
        "station_to_node_map": station_to_node, "station_names": top
    }
    results_path = RESULTS_FOLDER / f"simulation_results_seed_{seed_value}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results_to_save, f)
    print(f"✅ Results saved to {results_path}")


if __name__ == "__main__":
    seeds = [0, 11, 28]

    for seed in seeds:
        print(f"\n{'=' * 30} RUNNING SIMULATION FOR SEED: {seed} {'=' * 30}")

        # Defining seed-specific file path for the model
        model_path = RESULTS_FOLDER / f"models/DQN_Inventory_Model_Seed_{seed}.zip"

        run_simulation(
            inventory_model_path=model_path,
            graph_path=NYC_GRAPH_PATH,
            seed_value=seed
        )

    print(f"\n{'=' * 30} ALL SIMULATION RUNS COMPLETE {'=' * 30}")
