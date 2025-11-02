
import osmnx as ox, networkx as nx, geopandas as gpd
from .config import SEARCH_RADIUS_M

def _graph_type_for_mode(mode: str) -> str:
    m = mode.lower()
    if m in ("walk","walking","pedestrian","foot"): return "walk"
    if m in ("bike","bicycle","cycling"): return "bike"
    return "drive"

def download_graph(lat: float, lon: float, mode: str) -> nx.MultiDiGraph:
    gtype = _graph_type_for_mode(mode)
    dist  = SEARCH_RADIUS_M.get(mode, 1500)
    print(f"[INFO] Downloading OSM graph for mode='{mode}' radius={dist}m around ({lat:.6f},{lon:.6f})")
    G = ox.graph_from_point((lat, lon), dist=dist, network_type=gtype, simplify=True)
    print(f"[OK] Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

def nearest_node_no_sklearn(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    return int(ox.distance.nearest_nodes(G, X=lon, Y=lat))

def _edges_gdf_3414(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    Gp = ox.project_graph(G, to_crs=3414)
    result = ox.graph_to_gdfs(Gp, nodes=False, edges=True, fill_edge_geometry=True)
    g_edges = result[1] if isinstance(result, tuple) else result
    if g_edges.crs is None or (getattr(g_edges.crs, "to_epsg", lambda: None)() != 3414):
        g_edges = g_edges.set_crs(3414, allow_override=True)
    return g_edges
