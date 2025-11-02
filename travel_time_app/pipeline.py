
import osmnx as ox, networkx as nx, geopandas as gpd
from .models import POI
from .config import MODE_SPEEDS_KMH, NODE_PENALTY_MIN
from .geocoding import resolve_pois
from .graphs import download_graph, nearest_node_no_sklearn
from .routing import get_reachable_subgraph
from .polys import create_outer_boundary, export_boundary
from .plotting import plot_single_mode_frontiers, plot_multi_mode_overlay

def run_pipeline(location_name, poi_inputs, modes=("walk","bike","drive"), durations_min=(15,),
                 speeds_kmh=None, node_penalty_min=None, save_figs=True, export_vectors=False):
    speeds_kmh = speeds_kmh or MODE_SPEEDS_KMH
    node_penalty_min = NODE_PENALTY_MIN if node_penalty_min is None else node_penalty_min

    pois = resolve_pois(poi_inputs)
    if not pois:
        print("[ERROR] No valid POIs after geocoding."); return
    print(f"[INFO] Using {len(pois)} POIs: {[p.name for p in pois]}")

    per_mode_graphs = {m: [] for m in modes}
    for mode in modes:
        for poi in pois:
            G = download_graph(poi.lat, poi.lon, mode)
            per_mode_graphs[mode].append(G)

    all_graphs = [G for m in modes for G in per_mode_graphs[m]]

    nearest_nodes = {m: [] for m in modes}
    for mode in modes:
        for poi, G in zip(pois, per_mode_graphs[mode]):
            nearest_nodes[mode].append(nearest_node_no_sklearn(G, poi.lon, poi.lat))

    for dur in durations_min:
        print(f"[INFO] Computing frontiers for duration={dur} minutes")
        per_mode_polys = {m: [] for m in modes}
        for mode in modes:
            speed = speeds_kmh.get(mode, 5.0)
            mode_graphs = per_mode_graphs[mode]
            mode_nodes  = nearest_nodes[mode]
            mode_polys  = []
            for poi, G, center_node in zip(pois, mode_graphs, mode_nodes):
                subG = get_reachable_subgraph(G, center_node=center_node,
                                              travel_time_minutes=dur, speed_kmh=speed,
                                              node_penalty_min=node_penalty_min)
                gseries = create_outer_boundary(subG, buffer_extra_m=10.0)
                gseries.name = poi.name
                mode_polys.append(gseries)
                if export_vectors and (gseries is not None) and (len(gseries) > 0):
                    export_boundary(gseries, f"{location_name}_{mode}_{poi.name}_{int(dur)}min")
            per_mode_polys[mode] = mode_polys

            if save_figs:
                plot_single_mode_frontiers(location_name, mode, pois, mode_polys, mode_graphs,
                                           travel_time_minutes=dur, fade_steps=6, fade_max=0.25, save=True)

        if save_figs and len(modes) > 1:
            plot_multi_mode_overlay(location_name, pois, per_mode_polys, all_graphs,
                                    travel_time_minutes=dur, save=True, show_network=False)

    print("[OK] Pipeline complete.")
