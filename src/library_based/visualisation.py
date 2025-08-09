import json

import folium
import networkx as nx
import osmnx as ox
from folium.elements import Element
from folium.features import DivIcon
from folium.plugins import MarkerCluster
from geopy.distance import geodesic

from constants import BIKE_ICON_URL, TRUCK_ICON_URL


def plot_stations_cluster(cluster: MarkerCluster, coordinates: dict[str, tuple[float, float]],
                          bike_counts: dict[str, int], thresholds: dict[str, int]):
    """
    Plots station markers on a Folium map cluster.

    Args:
        cluster (MarkerCluster): The Folium marker cluster to add station markers to.
        coordinates (dict[str, tuple[float, float]]): Mapping of station names to their latitude and longitude.
        bike_counts (dict[str, int]): Mapping of station names to the current bike count.
        thresholds (dict[str, int]): Mapping of station names to threshold values for bike counts.

    Each marker displays the station name, bike count, and an icon. The marker color is green if the bike count
    meets or exceeds the threshold, otherwise orange.
    """
    for station, (lat, lon) in coordinates.items():
        cnt = bike_counts.get(station, 0)
        col = "green" if cnt >= thresholds.get(station, 0) else "orange"
        safe_name = station.replace(" ", "_").replace("&", "")
        html = (
            f"<div style='white-space:nowrap;text-align:center;'>"
            f"<span id='count_{safe_name}' style='background:{col};"
            f"color:white;padding:4px 6px;border-radius:4px;font-size:11px;'>"
            f"{station} 🚲 {cnt}</span>"
            f"<img src='{BIKE_ICON_URL}' width='24' height='24'/>"
            "</div>"
        )
        folium.Marker(location=[lat, lon], icon=DivIcon(icon_size=(150, 50), icon_anchor=(75, 25), html=html)).add_to(
            cluster)


def project_route_to_station(G, station_lat, station_lon):
    """
    Finds the nearest OSM node in the graph to the given station coordinates and computes the distance.

    Args:
        G (networkx.Graph): The OSMnx graph representing the road network.
        station_lat (float): Latitude of the station.
        station_lon (float): Longitude of the station.

    Returns:
        tuple: (node, nlat, nlon, dist)
            node (int): The nearest node ID in the graph.
            nlat (float): Latitude of the nearest node.
            nlon (float): Longitude of the nearest node.
            dist (float): Geodesic distance in meters between the station and the nearest node.
    """
    node = ox.nearest_nodes(G, station_lon, station_lat)
    nlat, nlon = G.nodes[node]['y'], G.nodes[node]['x']
    dist = geodesic((station_lat, station_lon), (nlat, nlon)).meters
    return node, nlat, nlon, dist


def draw_route(m, G, src_lat, src_lon, tgt_lat, tgt_lon, color="blue"):
    """
    Draws the shortest route between two geographic points on a Folium map.

    Args:
        m (folium.Map): The Folium map object to draw the route on.
        G (networkx.Graph): The OSMnx graph representing the road network.
        src_lat (float): Latitude of the source location.
        src_lon (float): Longitude of the source location.
        tgt_lat (float): Latitude of the target location.
        tgt_lon (float): Longitude of the target location.
        color (str, optional): Color of the route polyline. Defaults to "blue".

    Behavior:
        - Projects source and target coordinates to their nearest nodes in the graph.
        - Computes the shortest path between these nodes.
        - Adds intermediate points if the station is far from the nearest node.
        - Draws the route as a polyline on the map.
        - Handles exceptions and prints an error message if routing fails.
    """
    src_node, sn_lat, sn_lon, s_dist = project_route_to_station(G, src_lat, src_lon)
    tgt_node, tn_lat, tn_lon, t_dist = project_route_to_station(G, tgt_lat, tgt_lon)
    # Compute shortest path
    try:
        route_nodes = nx.shortest_path(G, src_node, tgt_node, weight="length")
        # Build route: station -> nearest node (if far), route, nearest node -> station (if far)
        points = []
        if s_dist > 50:
            points.append((src_lat, src_lon))  # Start at station
            points.append((sn_lat, sn_lon))  # Jump to road
        else:
            points.append((sn_lat, sn_lon))
        # Add all intermediate route nodes
        for n in route_nodes[1:-1]:
            points.append((G.nodes[n]['y'], G.nodes[n]['x']))
        if t_dist > 50:
            points.append((tn_lat, tn_lon))  # Off road to station
            points.append((tgt_lat, tgt_lon))
        else:
            points.append((tn_lat, tn_lon))
        folium.PolyLine(points, color=color, weight=5).add_to(m)
    except Exception as e:
        print(f"Routing failed: {e}")


def map_stations_to_nodes(G, stations_meta, lat_col="Latitude", lon_col="Longitude", warn_dist=200):
    """
    Maps each station to its nearest node in the OSMnx graph.

    Args:
        G (networkx.Graph): The OSMnx graph representing the road network.
        stations_meta (pd.DataFrame): DataFrame containing station metadata, indexed by station name.
        lat_col (str, optional): Name of the column containing latitude values. Defaults to "Latitude".
        lon_col (str, optional): Name of the column containing longitude values. Defaults to "Longitude".
        warn_dist (float, optional): Distance threshold (meters) to warn if a station is far from its nearest node. Defaults to 200.

    Returns:
        dict: Mapping of station names to their nearest node IDs in the graph.
    """
    station_to_node = {}
    for station in stations_meta.index:
        lat = float(stations_meta.loc[station, lat_col])
        lon = float(stations_meta.loc[station, lon_col])
        node = ox.distance.nearest_nodes(G, lon, lat)
        station_to_node[station] = node
    return station_to_node


def inject_animation_js(m: folium.Map, routes_data: list[dict], graph: nx.Graph) -> None:
    """
        Injects a JavaScript animation into a Folium map to visualize truck routes.

        Args:
            m (folium.Map): The Folium map object to inject the animation into.
            routes_data (list[dict]): List of route dictionaries, each containing:
                - truck_id (str): Identifier for the truck.
                - path_nodes (list[int]): Sequence of node IDs representing the route.
                - src_name (str): Source station name.
                - tgt_name (str): Target station name.
                - bikes_on_truck_start (int): Number of bikes on the truck at departure.
                - final_count_src (int): Final bike count at the source station.
                - final_count_tgt (int): Final bike count at the target station.
            graph (networkx.Graph): The OSMnx graph representing the road network.

        Behavior:
            - Animates truck markers moving along their routes on the map.
            - Updates bike counts at source and target stations.
            - Logs progress in a debug banner and the browser console.
            - Leaves truck markers at their final destination after animation.

        Returns:
            None
        """
    node_locs = {n: [graph.nodes[n]['y'], graph.nodes[n]['x']] for n in graph.nodes}

    payload = []
    for route in routes_data:
        path_coords = [node_locs[n] for n in route["path_nodes"] if n in node_locs]
        payload.append({
            "truckId": f"Truck_{route['truck_id']}",
            "path": path_coords,
            "duration": 10,
            "fromStation": route["src_name"].replace(" ", "_").replace("&", ""),
            "toStation": route["tgt_name"].replace(" ", "_").replace("&", ""),
            "bikesOnTruck": route["bikes_on_truck_start"],
            "finalCountSrc": route["final_count_src"],
            "finalCountTgt": route["final_count_tgt"],
        })

    js_data = json.dumps(payload)
    map_var = m.get_name()

    js_script = f"""
    <script>
    window.onload = async function () {{
      var map = window["{map_var}"];
      var routes = {js_data};

      var debugBanner = document.createElement('div');
      debugBanner.style.position = 'fixed'; debugBanner.style.bottom = '10px';
      debugBanner.style.left = '10px'; debugBanner.style.padding = '10px';
      debugBanner.style.background = 'rgba(0,0,0,0.7)'; debugBanner.style.color = 'white';
      debugBanner.style.zIndex = '9999'; debugBanner.style.fontFamily = 'monospace';
      debugBanner.innerText = '🚚 Animation Status:';
      document.body.appendChild(debugBanner);

      function animateTruck(leg) {{
        return new Promise(resolve => {{
            let marker = L.marker(leg.path[0], {{
                icon: L.divIcon({{
                    html: `<div style='text-align:center; white-space:nowrap;'>
                             <div style='background:black;color:white;padding:4px 8px;border-radius:4px;font-size:11px;'>${{leg.truckId}} 🚲 ${{leg.bikesOnTruck}}</div>
                             <img src='{TRUCK_ICON_URL}' width='30' height='30'/>
                           </div>`,
                    className: '', iconSize: [90, 40], iconAnchor: [45, 40]
                }})
            }}).addTo(map);

            let departMsg = `▶️ ${{leg.truckId}}: Departing ${{leg.fromStation.replace('_', ' ')}} (${{leg.bikesOnTruck}} bikes)`;
            debugBanner.innerText += `\\n${{departMsg}}`;
            console.log(departMsg);

            const srcElem = document.getElementById('count_' + leg.fromStation);
            if(srcElem) {{ srcElem.innerText = `${{leg.fromStation.replace('_',' ')}} 🚲 ${{leg.finalCountSrc}}`; }}

            let i = 0;
            const steps = leg.path.length;
            const delay = (leg.duration * 1000) / steps;

            function move() {{
                if (i < steps) {{
                    marker.setLatLng(leg.path[i]); i++; setTimeout(move, delay);
                }} else {{
                    // THIS IS THE CORRECTED LINE
                    const tgtElem = document.getElementById('count_' + leg.toStation);

                    if (tgtElem) {{
                        tgtElem.innerText = `${{leg.toStation.replace('_',' ')}} 🚲 ${{leg.finalCountTgt}}`;
                        tgtElem.style.background = 'green';
                    }}
                    let arriveMsg = `✅ ${{leg.truckId}}: Arrived at ${{leg.toStation.replace('_', ' ')}}`;
                    debugBanner.innerText += `\\n${{arriveMsg}}`;
                    console.log(arriveMsg);
                    resolve();
                }}
            }}
            move();
        }});
      }}

      const trucks = routes.reduce((acc, route) => {{
        acc[route.truckId] = acc[route.truckId] || [];
        acc[route.truckId].push(route);
        return acc;
      }}, {{}});

      for (const truckId in trucks) {{
        let dispatchMsg = `--- Dispatching ${{truckId}} ---`;
        debugBanner.innerText += `\\n\\n${{dispatchMsg}}`;
        console.log(`\\n${{dispatchMsg}}`);
        for (const leg of trucks[truckId]) {{
            await animateTruck(leg);
        }}
      }}
    }};
    </script>
    """
    m.get_root().html.add_child(Element(js_script))
