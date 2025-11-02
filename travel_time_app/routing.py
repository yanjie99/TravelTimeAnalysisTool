
import networkx as nx

def _ensure_time_weights(G: nx.MultiDiGraph, speed_kmh: float, node_penalty_min: float):
    m_per_min = speed_kmh * 1000.0 / 60.0
    for _, _, _, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        if not (length_m > 0):
            length_m = 5.0  # clamp tiny edges
        data["time"] = length_m / m_per_min if m_per_min > 0 else 1e9
    if node_penalty_min and node_penalty_min > 0:
        for n in G.nodes():
            out_edges = list(G.out_edges(n, keys=True, data=True))
            deg = len(out_edges)
            if deg > 0:
                add_each = node_penalty_min / deg
                for _, _, _, data in out_edges:
                    data["time"] = float(data.get("time", 0.0)) + add_each

def get_reachable_subgraph(G: nx.MultiDiGraph, center_node: int, travel_time_minutes: float,
                           speed_kmh: float, node_penalty_min: float) -> nx.MultiDiGraph:
    _ensure_time_weights(G, speed_kmh=speed_kmh, node_penalty_min=node_penalty_min)
    lengths = nx.single_source_dijkstra_path_length(G, center_node, cutoff=travel_time_minutes, weight="time")
    nodes_reached = [n for n, t in lengths.items() if t <= travel_time_minutes]
    return G.subgraph(nodes_reached).copy() if nodes_reached else G.subgraph([]).copy()
