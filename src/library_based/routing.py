import logging
from datetime import datetime, timedelta

import networkx as nx


# ==========================================================
# Parallel Pathfinding
# ==========================================================

def init_worker(graph, station_nodes):
    """
    Initializer for each worker process with the graph data.

    Args:
        graph (networkx.Graph): The graph representing the network.
        station_nodes (dict): Mapping from station names to node identifiers.

    Sets:
        G_worker: Global variable for the worker's graph.
        station_to_node_worker: Global variable for station-to-node mapping.
    """
    global G_worker, station_to_node_worker
    G_worker = graph
    station_to_node_worker = station_nodes


def find_path_worker(task):
    """
    Finds the shortest path between two stations using the globally available graph.

    Args:
        task (tuple): A tuple containing the source and target station names.

    Returns:
        tuple: (src, tgt, path_nodes) where path_nodes is a list of node identifiers representing the shortest path,
               or (src, tgt, None) if no path is found.
    """
    src, tgt = task
    try:
        path_nodes = nx.shortest_path(G_worker, station_to_node_worker[src], station_to_node_worker[tgt],
                                      weight="length")
        return (src, tgt, path_nodes)
    except Exception as e:
        logging.warning(f"Parallel pathfinding failed for {src}->{tgt}: {e}")
        return (src, tgt, None)


def plan_optimized_journeys(initial_counts: dict, thresholds: dict, station_names: list) -> tuple:
    """
    Plans efficient, multi-leg journeys for a fleet of trucks to redistribute bikes between stations.

    Args:
        initial_counts (dict): Current bike counts at each station, keyed by station name.
        thresholds (dict): Target bike counts for each station, keyed by station name.
        station_names (list): List of all station names.

    Returns:
        tuple:
            - optimized_journeys (list): List of journey dicts, each with truck_id and legs (source, target, move count).
            - live_counts (dict): Final bike counts at each station after all journeys.
    """
    print(f"\nStep 4: Starting journey planning...")
    live_counts = initial_counts.copy()
    surplus_stations = [s for s in station_names if live_counts[s] > thresholds[s]]
    optimized_journeys = []
    truck_counter = 0

    while surplus_stations:
        src_name = max(surplus_stations, key=lambda s: live_counts[s] - thresholds[s])
        surplus_stations.remove(src_name)

        journey = {"truck_id": truck_counter, "legs": []}
        current_location = src_name

        print(f"\n🚚 Planning journey for Truck_{truck_counter}, starting from '{src_name}'...")

        while True:
            surplus = live_counts[current_location] - thresholds.get(current_location, 0)
            if surplus <= 0:
                print(f"   - Journey for Truck_{truck_counter} complete. No more surplus at '{current_location}'.")
                break

            bikes_on_truck = surplus
            live_counts[current_location] -= bikes_on_truck
            print(f"   Picking up {bikes_on_truck} bikes at '{current_location}'.")

            best_next_tgt = None
            for potential_tgt in station_names:
                if potential_tgt != current_location and live_counts[potential_tgt] < thresholds[potential_tgt]:
                    best_next_tgt = potential_tgt
                    break

            if not best_next_tgt:
                live_counts[current_location] += bikes_on_truck
                print(f"   - Journey for Truck_{truck_counter} complete. No valid destination found.")
                break

            needed = thresholds[best_next_tgt] - live_counts[best_next_tgt]
            drop_count = min(bikes_on_truck, needed)

            leg = {"src": current_location, "tgt": best_next_tgt, "move": drop_count}
            journey["legs"].append(leg)
            print(f"   -> Leg planned: {current_location} to {best_next_tgt} ({drop_count} bikes)")

            live_counts[best_next_tgt] += drop_count
            bikes_on_truck -= drop_count
            current_location = best_next_tgt
            if bikes_on_truck > 0:
                live_counts[current_location] += bikes_on_truck

        if journey["legs"]:
            optimized_journeys.append(journey)
            truck_counter += 1

    print("\n✅ All truck journeys planned successfully.")
    return optimized_journeys, live_counts


def schedule_journeys(optimized_journeys: list, transfers: dict, station_names: list) -> list:
    """
    Assigns a proactive, just-in-time dispatch schedule to a list of planned journeys.

    Args:
        optimized_journeys (list): List of journey dicts, each with truck_id and legs (source, target, move count).
        transfers (dict): Mapping of (src_idx, tgt_idx) to transfer details, including 'first_hour'.
        station_names (list): List of all station names.

    Returns:
        list: The input journeys, each with scheduled dispatch times for every leg and an 'urgency_hour' field.
    """
    print("\nStep 5: Scheduling parallel, just-in-time dispatches...")

    if not optimized_journeys:
        print("   - No journeys to schedule.")
        return optimized_journeys

    for journey in optimized_journeys:
        first_leg = journey['legs'][0]
        src_idx, tgt_idx = station_names.index(first_leg['src']), station_names.index(first_leg['tgt'])
        original_transfer = transfers.get((src_idx, tgt_idx))
        journey['urgency_hour'] = original_transfer['first_hour'] if original_transfer else 23

    optimized_journeys.sort(key=lambda j: j['urgency_hour'])

    truck_availability_time = {}
    estimated_leg_duration = timedelta(minutes=30)

    for journey in optimized_journeys:
        truck_id = journey['truck_id']
        earliest_need_hour = journey['urgency_hour']

        arrival_time = datetime.strptime(f"{earliest_need_hour}:00", "%H:%M") - timedelta(hours=1)
        dispatch_time = arrival_time - estimated_leg_duration

        truck_free_at = truck_availability_time.get(truck_id, datetime.strptime("00:00", "%H:%M"))
        actual_dispatch_time = max(dispatch_time, truck_free_at)

        time_cursor = actual_dispatch_time
        for leg in journey['legs']:
            leg['dispatch_time'] = time_cursor.strftime("%I:%M %p")
            time_cursor += estimated_leg_duration

        truck_availability_time[truck_id] = time_cursor

    print("   - ✅ All tasks have been scheduled.")
    return optimized_journeys
