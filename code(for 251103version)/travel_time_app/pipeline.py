import osmnx as ox, networkx as nx, geopandas as gpd
from typing import Callable, Optional, Iterable, Dict, List
from .models import POI
from .config import MODE_SPEEDS_KMH, NODE_PENALTY_MIN
from .geocoding import resolve_pois
from .graphs import download_graph, nearest_node_no_sklearn
from .routing import get_reachable_subgraph
from .polys import create_outer_boundary, export_boundary
from .plotting import plot_single_mode_frontiers, plot_multi_mode_overlay

ProgressCB = Optional[Callable[[int, str], None]]

def _emit(progress: ProgressCB, pct: int, msg: str):
    if progress:
        try:
            progress(int(max(0, min(100, pct))), msg)
        except Exception:
            pass

def run_pipeline(location_name: str,
                 poi_inputs: Iterable,
                 modes: Iterable[str] = ("walk","bike","drive"),
                 durations_min: Iterable[float] = (15,),
                 speeds_kmh: Optional[Dict[str, float]] = None,
                 node_penalty_min: Optional[float] = None,
                 save_figs: bool = True,
                 export_vectors: bool = True,
                 progress: ProgressCB = None) -> None:
    """Pipeline with optional `progress(pct:int, msg:str)` callback."""
    speeds_kmh = speeds_kmh or MODE_SPEEDS_KMH
    node_penalty_min = NODE_PENALTY_MIN if node_penalty_min is None else node_penalty_min

    _emit(progress, 1, "Resolving POIs")
    pois = resolve_pois(poi_inputs)
    if not pois:
        _emit(progress, 0, "No valid POIs after geocoding")
        print("[ERROR] No valid POIs after geocoding."); return
    print(f"[INFO] Using {len(pois)} POIs: {[p.name for p in pois]}")

    pois_n   = len(pois)
    modes_l  = list(modes)
    modes_n  = len(modes_l)
    durs_l   = list(durations_min)
    durs_n   = len(durs_l)

    steps_geo   = pois_n
    steps_dl    = modes_n * pois_n
    steps_nn    = modes_n * pois_n
    steps_route = durs_n * modes_n * pois_n
    steps_poly  = durs_n * modes_n * pois_n
    steps_vec   = durs_n * modes_n * pois_n if export_vectors else 0
    steps_plot  = durs_n * modes_n
    steps_multi = durs_n * (1 if modes_n > 1 else 0)

    total_steps = max(1, steps_geo + steps_dl + steps_nn + steps_route + steps_poly + steps_vec + steps_plot + steps_multi)
    done = 0
    def tick(msg):
        nonlocal done
        done += 1
        _emit(progress, int(100 * done / total_steps), msg)

    _emit(progress, 5, "Downloading OSM graphs")
    per_mode_graphs: Dict[str, List[nx.MultiDiGraph]] = {m: [] for m in modes_l}
    for mode in modes_l:
        for poi in pois:
            G = download_graph(poi.lat, poi.lon, mode)
            per_mode_graphs[mode].append(G)
            tick(f"Downloaded graph: {mode} @ {poi.name}")

    _emit(progress, 15, "Locating nearest nodes")
    nearest_nodes: Dict[str, List[int]] = {m: [] for m in modes_l}
    for mode in modes_l:
        for poi, G in zip(pois, per_mode_graphs[mode]):
            node = nearest_node_no_sklearn(G, poi.lon, poi.lat)
            nearest_nodes[mode].append(node)
            tick(f"Nearest node: {mode} @ {poi.name}")

    _emit(progress, 25, "Routing & building boundaries")
    for dur in durs_l:
        print(f"[INFO] Computing frontiers for duration={dur} minutes")
        per_mode_polys: Dict[str, List[gpd.GeoSeries]] = {m: [] for m in modes_l}

        for mode in modes_l:
            speed = (speeds_kmh.get(mode, 5.0) if isinstance(speeds_kmh, dict) else 5.0)
            mode_graphs = per_mode_graphs[mode]
            mode_nodes  = nearest_nodes[mode]
            mode_polys  = []

            for poi, G, center_node in zip(pois, mode_graphs, mode_nodes):
                subG = get_reachable_subgraph(
                    G, center_node=center_node,
                    travel_time_minutes=dur,
                    speed_kmh=speed,
                    node_penalty_min=node_penalty_min
                )
                tick(f"Routed: {mode} @ {poi.name} ({dur}m)")

                gseries = create_outer_boundary(subG, buffer_extra_m=10.0)
                gseries.name = poi.name
                mode_polys.append(gseries)
                tick(f"Boundary: {mode} @ {poi.name} ({dur}m)")

                if export_vectors and (gseries is not None) and (len(gseries) > 0):
                    base_name = f"{location_name}_{mode}_{poi.name}_{int(dur)}min"
                    export_boundary(gseries, base_name)
                    tick(f"Exported: {mode} @ {poi.name} ({dur}m)")

            per_mode_polys[mode] = mode_polys

            if save_figs:
                plot_single_mode_frontiers(
                    location_name=location_name,
                    mode=mode,
                    pois=pois,
                    poi_polys=mode_polys,
                    base_graphs=mode_graphs,
                    travel_time_minutes=dur,
                    fade_steps=6,
                    fade_max=0.25,
                    save=True
                )
                tick(f"Plotted: {mode} ({dur}m)")

        if save_figs and len(modes_l) > 1:
            plot_multi_mode_overlay(
                location_name=location_name,
                pois=pois,
                per_mode_polys=per_mode_polys,
                base_graphs_all=[G for m in modes_l for G in per_mode_graphs[m]],
                travel_time_minutes=dur,
                save=True,
                show_network=False
            )
            tick(f"Plotted: multi-mode ({dur}m)")

    _emit(progress, 100, "Done")
    print("[OK] Pipeline complete.")
