import folium
import pandas as pd
from folium.plugins import MarkerCluster

from constants import TARGET_DATE, TOP_N
from smartflow import calculate_thresholds, DATA_FILE, find_dual_policy_routes_all, get_station_coordinates, \
    inject_animation_js, load_q_tables, MAP_CENTER, MAP_ZOOM, plot_stations_cluster, simulate_bike_counts

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)
    df = df[df["Start Time"].notna()]
    df["Start Time"] = pd.to_datetime(df["Start Time"])
    df = df[df["Start Time"].dt.date == TARGET_DATE]
    top_stations = df["Start Station Name"].value_counts().index.tolist()[:TOP_N]
    trip_counts = df['Start Station Name'].value_counts()
    thresholds = calculate_thresholds(trip_counts)

    q_source, q_target = load_q_tables()
    coordinates = get_station_coordinates(df)
    coordinates = {s: coordinates[s] for s in top_stations if s in coordinates}
    bike_counts = simulate_bike_counts(thresholds)
    routes, transfers = find_dual_policy_routes_all(top_stations, coordinates, q_source, q_target, bike_counts,
                                                    thresholds)

    folium_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    marker_cluster = MarkerCluster().add_to(folium_map)
    plot_stations_cluster(marker_cluster, coordinates, bike_counts, thresholds)
    inject_animation_js(folium_map, coordinates, routes, transfers, bike_counts)
    folium_map.save("../../results/simulation/oneoff_simulation.html")
    print("Map saved to oneoff_simulation.html")
