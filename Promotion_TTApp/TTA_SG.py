# ============================================================
# 15-Minute City Travel Frontiers (Singapore-ready, generalizable)
# Author: Yanjie Zhang & ChatGPT5 (2025.10.30 version)
# Full pipeline with professional plotting, basemap, and labels.
# Compatible with osmnx 1.x and 2.x.
# ============================================================
# Avoid map display separately
import matplotlib
matplotlib.use("Agg", force=True)

from dataclasses import dataclass
from typing import List, Dict, Iterable, Optional, Tuple
import time
import os

# Core geo+graph stack
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import contextily as cx

# -------------------------------
# Global configuration
# -------------------------------
MODE_SPEEDS_KMH: Dict[str, float] = {
    "walk": 5.0,
    "bike": 15.0,
    "drive": 35.0,   # tune for your context / congestion
}

# Search radius (meters) around each POI to fetch the OSM graph
SEARCH_RADIUS_M: Dict[str, int] = {
    "walk": 1500,
    "bike": 6250,
    "drive": 10000,
}

# A tiny per-node penalty (minutes) to discourage unrealistically zig-zaggy paths
NODE_PENALTY_MIN: float = 0.02

# Output directories
MAP_DIR = "maps"
VEC_DIR = "frontiers"

os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(VEC_DIR, exist_ok=True)

# Professional mode styles
MODE_STYLE = {
    "walk":  {"color": "#44DD88", "label": "Walk"},
    "bike":  {"color": "#FF9933", "label": "Bike"},
    "drive": {"color": "#4F9DFF", "label": "Drive"},
}



TITLE_FONTS = dict(fontsize=14, fontweight="bold")
SUBTITLE_FONTS = dict(fontsize=10, color="#555")

# -------------------------------
# Data model
# -------------------------------
@dataclass
class POI:
    name: str
    lat: float
    lon: float

# -------------------------------
# Utilities
# -------------------------------
def SAFE(s: str) -> str:
    """Safe string for filenames."""
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s).strip("_")

def _graph_type_for_mode(mode: str) -> str:
    m = mode.lower()
    if m in ("walk", "walking", "pedestrian", "foot"):
        return "walk"
    if m in ("bike", "bicycle", "cycling"):
        return "bike"
    return "drive"

def minutes_per_meter(speed_kmh: float) -> float:
    speed_m_per_min = speed_kmh * 1000.0 / 60.0
    return 1.0 / speed_m_per_min

# -------------------------------
# Geocoding
# -------------------------------
def geocode_freeform(q: str, retries: int = 3, pause: float = 1.0) -> Optional[POI]:
    """
    Geocode a freeform query with osmnx (Nominatim).
    Returns POI or None if not found.
    """
    for i in range(retries):
        try:
            # osmnx 1.x/2.x: ox.geocoder.geocode returns (lat, lon) or shapely object.
            res = ox.geocoder.geocode(q)
            if isinstance(res, (list, tuple)) and len(res) == 2 and all(isinstance(x, (int, float)) for x in res):
                lat, lon = float(res[0]), float(res[1])
            else:
                # if geometry returned, take centroid
                geom = gpd.GeoSeries([res], crs=4326)
                lon, lat = geom.to_crs(4326).geometry.iloc[0].centroid.coords[0]
            print(f"[OK] Geocoded '{q}' -> ({lat:.6f}, {lon:.6f})")
            return POI(name=q, lat=lat, lon=lon)
        except Exception as e:
            print(f"[WARN] Geocode attempt {i+1} failed for '{q}': {e}")
            time.sleep(pause)
    print(f"[ERROR] Failed to geocode '{q}'")
    return None

def resolve_pois(poi_inputs: Iterable) -> List[POI]:
    """
    Accepts:
      - List[str] of names to geocode, or
      - List[POI]
    Returns a clean list[POI].
    """
    out: List[POI] = []
    for item in poi_inputs:
        if isinstance(item, POI):
            out.append(item)
        elif isinstance(item, str):
            p = geocode_freeform(item)
            if p:
                out.append(p)
        else:
            print(f"[WARN] Unsupported POI input: {item!r}")
    return out

# -------------------------------
# Graph build & nearest nodes
# -------------------------------
def download_graph(lat: float, lon: float, mode: str) -> nx.MultiDiGraph:
    gtype = _graph_type_for_mode(mode)
    dist = SEARCH_RADIUS_M.get(mode, 1500)
    print(f"[INFO] Downloading OSM graph for mode='{mode}' radius={dist}m around ({lat:.6f},{lon:.6f})")
    G = ox.graph_from_point((lat, lon), dist=dist, network_type=gtype, simplify=True)
    print(f"[OK] Graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G

def nearest_node_no_sklearn(G: nx.MultiDiGraph, lon: float, lat: float) -> int:
    """
    Find nearest node in graph (graph x/y are lon/lat in EPSG:4326).
    """
    node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    return int(node)

# -------------------------------
# Travel time weighting & reachability
# -------------------------------
def _ensure_time_weights(G: nx.MultiDiGraph, speed_kmh: float, node_penalty_min: float = NODE_PENALTY_MIN) -> None:
    """
    Add 'time' (minutes) to each edge based on length and speed.
    Distribute a tiny node penalty across outgoing edges for realism.
    """
    m_per_min = speed_kmh * 1000.0 / 60.0
    # Edge travel time
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        t_edge = length_m / m_per_min if m_per_min > 0 else 1e9
        data["time"] = t_edge

    # Node penalty distribution
    if node_penalty_min and node_penalty_min > 0:
        for n in G.nodes():
            out_edges = list(G.out_edges(n, keys=True, data=True))
            deg = len(out_edges)
            if deg > 0:
                add_each = node_penalty_min / deg
                for _, _, _, data in out_edges:
                    data["time"] = float(data.get("time", 0.0)) + add_each

def get_reachable_subgraph(
    G: nx.MultiDiGraph,
    center_node: int,
    travel_time_minutes: float,
    speed_kmh: float,
    node_penalty_min: float = NODE_PENALTY_MIN
) -> nx.MultiDiGraph:
    """
    Returns the induced subgraph of nodes reachable within the time budget (minutes).
    """
    _ensure_time_weights(G, speed_kmh=speed_kmh, node_penalty_min=node_penalty_min)
    lengths = nx.single_source_dijkstra_path_length(G, center_node, cutoff=travel_time_minutes, weight="time")
    nodes_reached = [n for n, t in lengths.items() if t <= travel_time_minutes]
    if not nodes_reached:
        return G.subgraph([]).copy()
    return G.subgraph(nodes_reached).copy()

# -------------------------------
# Boundary polygon from reachable subgraph
# -------------------------------
def _edges_gdf_3857(G: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Return edges GeoDataFrame in EPSG:3414 (SVY21 / Singapore TM),
    compatible with osmnx 1.x/2.x.
    """
    # Project to 3414 for meter-accurate geometry in Singapore
    Gp = ox.project_graph(G, to_crs=3414)
    result = ox.graph_to_gdfs(Gp, nodes=False, edges=True, fill_edge_geometry=True)
    if isinstance(result, tuple):
        g_edges = result[1]
    else:
        g_edges = result
    # Ensure CRS is 3414
    if g_edges.crs is None or int(getattr(g_edges.crs, "to_epsg", lambda: None)() or 0) != 3414:
        g_edges = g_edges.set_crs(3414, allow_override=True)
    return g_edges
    
def create_outer_boundary(reachable_subgraph: nx.MultiDiGraph, buffer_extra_m: float = 10.0) -> gpd.GeoSeries:
    """
    Convert reachable edges into a single buffered polygon frontier (EPSG:3414).
    Returns empty GeoSeries (EPSG:3414) if nothing reachable.
    """
    empty_3414 = gpd.GeoSeries([], dtype=object, crs=3414)

    if reachable_subgraph is None or reachable_subgraph.number_of_edges() == 0:
        return empty_3414

    # Edges now in 3414 (despite helper name)
    edges3414 = _edges_gdf_3857(reachable_subgraph)
    if edges3414.empty:
        return empty_3414

    # Adaptive buffer in meters (3414)
    lengths = edges3414.length.values
    mean_len = float(np.nanmean(lengths)) if len(lengths) else 20.0
    base = max(5.0, min(60.0, mean_len * 0.3))
    buf_dist = base + float(buffer_extra_m)

    # Dissolve to one multilinestring, then buffer once (faster/cleaner)
    lines = unary_union(edges3414.geometry.values)
    if lines.is_empty:
        return empty_3414

    buffered = lines.buffer(buf_dist, cap_style=2, join_style=2)

    # Normalize to a single polygon
    poly: Optional[Polygon] = None
    if isinstance(buffered, Polygon):
        poly = buffered
    elif isinstance(buffered, MultiPolygon):
        if len(buffered.geoms) == 0:
            return empty_3414
        poly = max(buffered.geoms, key=lambda p: p.area)
    else:
        try:
            polys = [g for g in getattr(buffered, "geoms", []) if isinstance(g, Polygon)]
            if not polys:
                return empty_3414
            poly = max(polys, key=lambda p: p.area)
        except Exception:
            return empty_3414

    # Return in EPSG:3414
    gseries_3414 = gpd.GeoSeries([poly], crs=3414)
    return gseries_3414


def export_boundary(gseries: gpd.GeoSeries, base_name: str) -> None:
    if gseries is None or len(gseries) == 0:
        print(f"[WARN] Nothing to export for {base_name}")
        return

    # --- normalize to EPSG:4326 for export ---
    gs = gseries
    crs_in = getattr(gs, "crs", None)

    if crs_in is None:
        # Heuristic: if coords look projected (big numbers), assume 3414; else 4326
        try:
            # compute bounds without relying on CRS
            minx, miny, maxx, maxy = gpd.GeoSeries(gs, dtype=object).total_bounds
            projected_like = (max(abs(minx), abs(maxx)) > 500) or (max(abs(miny), abs(maxy)) > 500)
            assumed_crs = 3414 if projected_like else 4326
            gs = gpd.GeoSeries(gs, crs=assumed_crs)
        except Exception:
            gs = gpd.GeoSeries(gs, crs=4326)  # safest default

    gs_4326 = gs.to_crs(4326) if gs.crs != 4326 else gs
    # -----------------------------------------

    safe = SAFE(base_name)
    os.makedirs(VEC_DIR, exist_ok=True)

    gdf = gpd.GeoDataFrame({"name": [base_name]}, geometry=gs_4326, crs=4326)
    out_geojson = os.path.join(VEC_DIR, f"{safe}.geojson")
    out_gpkg   = os.path.join(VEC_DIR, f"{safe}.gpkg")
    gdf.to_file(out_geojson, driver="GeoJSON")
    gdf.to_file(out_gpkg, driver="GPKG")
    print(f"[OK] Exported: {out_geojson}  |  {out_gpkg}")


# -------------------------------
# Plot helpers
# -------------------------------
def _graphs_extent_3857(graphs: List[nx.MultiDiGraph]) -> Tuple[float, float, float, float]:
    polys = []
    for G in graphs:
        edges = _edges_gdf_3857(G)   # returns 3414 now
        if not edges.empty:
            polys.append(edges.unary_union.envelope)
    if polys:
        union = unary_union(polys)
        minx, miny, maxx, maxy = union.bounds
    else:
        # Fallback around Singapore in 3414 (NOT 3857)
        fallback = gpd.GeoSeries([Point(103.8198, 1.3521)], crs=4326).to_crs(3414).buffer(15_000).total_bounds
        minx, miny, maxx, maxy = fallback
    dx, dy = (maxx - minx) * 0.06, (maxy - miny) * 0.06
    # keep the return ordering your code expects
    return (minx - dx, maxx + dx, miny - dy, maxy + dy)

def _add_basemap(ax, zoom="auto"):
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.DarkMatter,
        crs="EPSG:3414",     # axis/data CRS
        zoom=zoom,
        reset_extent=False   # <-- keep the current x/y limits
    )


def _format_duration(tt) -> str:
    return f"{int(tt)} min" if float(tt).is_integer() else f"{tt:.1f} min"

def _draw_scale_bar(ax, extent_3857, loc="lower right", length_m=1000):
    minx, maxx, miny, maxy = extent_3857
    pad_x = (maxx - minx) * 0.04
    pad_y = (maxy - miny) * 0.04
    y = miny + pad_y if "lower" in loc else maxy - pad_y
    x_right = maxx - pad_x if "right" in loc else minx + pad_x
    x0 = x_right - length_m
    ax.plot([x0, x_right], [y, y], lw=3, solid_capstyle="butt", color="#FFFFFF")
    ax.text(
        (x0 + x_right) / 2, y + pad_y * 0.25,
        f"{length_m/1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m",
        ha="center", va="bottom", fontsize=9, color="#222",
        bbox=dict(fc="white", ec="none", alpha=0.7, pad=2)
    )


def _plot_pois_with_labels(ax, pois_4326: List[POI]) -> gpd.GeoDataFrame:
    poi_gdf = gpd.GeoDataFrame(
        {"name": [p.name for p in pois_4326]},
        geometry=[Point(p.lon, p.lat) for p in pois_4326],
        crs=4326
    ).to_crs(3414)  # <<< was 3857
    if not poi_gdf.empty:
        poi_gdf.plot(ax=ax, markersize=36, color="white", edgecolor="#111", zorder=6)
        for _, r in poi_gdf.iterrows():
            ax.text(
                r.geometry.x, r.geometry.y, r["name"],
                ha="left", va="bottom", fontsize=9, color="#111",
                bbox=dict(fc="white", ec="#CCC", alpha=0.85, boxstyle="round,pad=0.2"),
                zorder=7
            )
    return poi_gdf



def _draw_polygon(ax, poly: Polygon, facecolor: str, edgecolor: str, alpha_fill: float, lw: float = 1.4):
    coords = np.asarray(poly.exterior.coords)
    ax.add_patch(
        plt.Polygon(
            coords,
            fill=True, facecolor=facecolor, alpha=alpha_fill,
            edgecolor=edgecolor, lw=lw
        )
    )

# -------------------------------
# Plot: single-mode frontiers
# -------------------------------
# def plot_single_mode_frontiers(
#     location_name: str,
#     mode: str,
#     pois: List[POI],              # real names + coords
#     poi_polys: List[gpd.GeoSeries],  # one GeoSeries per POI, EPSG:4326
#     base_graphs: List[nx.MultiDiGraph],
#     travel_time_minutes: float,
#     fade_steps: int = 6,
#     fade_max: float = 0.25,
#     save: bool = True
# ):
#     extent = _graphs_extent_3857(base_graphs)
#     fig, ax = plt.subplots(figsize=(10, 9), dpi=150)

#     # _add_basemap(ax)

#     # Faint base networks
#     for G in base_graphs:
#         edges = _edges_gdf_3857(G)
#         if not edges.empty:
#             edges.plot(ax=ax, lw=0.6, color="#BBB", alpha=0.6)

#     style = MODE_STYLE.get(mode, {"color": "#777", "label": mode.title()})
#     color = style["color"]

#     # Frontiers with fade effect
#     for gseries in poi_polys:
#         if gseries is None or len(gseries) == 0:
#             continue
#         crs_in = getattr(gseries, "crs", None) or 4326
#         gdf3414 = gpd.GeoDataFrame(geometry=gseries).set_crs(crs_in, allow_override=True).to_crs(3414)
#         geom = gdf3414.geometry.iloc[0]
#         polys: List[Polygon] = []
#         if isinstance(geom, Polygon):
#             polys = [geom]
#         elif isinstance(geom, MultiPolygon):
#             polys = list(geom.geoms)

#         for poly in polys:
#             # Fade rings (outer → inner)
#             for i in range(fade_steps, 0, -1):
#                 alpha = fade_max * (i / fade_steps)
#                 _draw_polygon(ax, poly, facecolor=color, edgecolor=color, alpha_fill=alpha, lw=0.8)
#             # Crisp outline
#             _draw_polygon(ax, poly, facecolor="none", edgecolor=color, alpha_fill=0.35, lw=1.6)

#     # POIs and labels
#     _plot_pois_with_labels(ax, pois)

#     ax.set_title(f"{location_name}: {style['label']} {_format_duration(travel_time_minutes)} Frontier", **TITLE_FONTS)
#     ax.text(0.01, 0.98, "Reachable area by travel time • OSM network • Tiles © CartoDB",
#             transform=ax.transAxes, ha="left", va="top", **SUBTITLE_FONTS)

#     # Legend
#     handles = [
#         Patch(facecolor=color, edgecolor=color, alpha=0.6, label=f"{style['label']} frontier"),
#         Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor="#111", markersize=8, label="POI"),
#     ]
#     ax.legend(handles=handles, frameon=True, framealpha=0.92, loc="upper right")

#     ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    
#     # Add basemap AFTER limits are final
#     _add_basemap(ax)
    
#     _draw_scale_bar(ax, extent, loc="lower right", length_m=1000)
#     ax.set_axis_off()

#     if save:
#         out = os.path.join(MAP_DIR, f"{SAFE(location_name)}_{mode}_{int(travel_time_minutes)}min.png")
#         plt.tight_layout()
#         plt.savefig(out, bbox_inches="tight", dpi=200)
#         print(f"[OK] Saved map: {out}")
#     plt.close(fig)

# def plot_multi_mode_overlay(
#     location_name: str,
#     pois: List[POI],
#     per_mode_polys: Dict[str, List[gpd.GeoSeries]],   # mode -> list of GeoSeries
#     base_graphs_all: List[nx.MultiDiGraph],
#     travel_time_minutes: float,
#     save: bool = True
# ):
#     extent = _graphs_extent_3857(base_graphs_all)
#     fig, ax = plt.subplots(figsize=(10, 9), dpi=150)

#     # _add_basemap(ax)

#     # Base networks
#     for G in base_graphs_all:
#         edges = _edges_gdf_3857(G)
#         if not edges.empty:
#             edges.plot(ax=ax, lw=0.5, color="#BDBDBD", alpha=0.6)

#     # ------- Per-mode polygons with explicit bottom→top order -------
#     draw_order = ["drive", "bike", "walk"]
#     # include any unexpected modes after the preferred ones (defensive)
#     draw_order += [m for m in per_mode_polys.keys() if m not in draw_order]

#     handles = []
#     for mode in draw_order:
#         if mode not in per_mode_polys:
#             continue
#         style = MODE_STYLE.get(mode, {"color": "#777", "label": mode.title()})
#         color = style["color"]
#         any_drawn = False

#         for gseries in per_mode_polys[mode]:
#             if gseries is None or len(gseries) == 0:
#                 continue
#             crs_in = getattr(gseries, "crs", None) or 4326
#             gdf3414 = gpd.GeoDataFrame(geometry=gseries).set_crs(crs_in, allow_override=True).to_crs(3414)
#             geom = gdf3414.geometry.iloc[0]

#             if isinstance(geom, Polygon):
#                 polys = [geom]
#             elif isinstance(geom, MultiPolygon):
#                 polys = list(geom.geoms)
#             else:
#                 polys = []

#             for poly in polys:
#                 _draw_polygon(ax, poly, facecolor=color, edgecolor=color, alpha_fill=0.35, lw=1.4)
#                 any_drawn = True

#         if any_drawn:
#             handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=style["label"]))
#     # ----------------------------------------------------------------

#     # POIs and labels (+legend handle)
#     _plot_pois_with_labels(ax, pois)
#     handles.append(Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor="#111",
#                           markersize=8, label="POI"))

#     ax.set_title(f"{location_name}: Multi-mode {_format_duration(travel_time_minutes)} Frontiers", **TITLE_FONTS)
#     ax.text(0.01, 0.98, "Reachable areas by mode and travel time • OSM network • Tiles © CartoDB",
#             transform=ax.transAxes, ha="left", va="top", **SUBTITLE_FONTS)

#     if handles:
#         # legend follows the same visual stacking order
#         ax.legend(handles=handles, frameon=True, framealpha=0.92, loc="upper right")

#     ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

#     # Add basemap AFTER limits are final
#     _add_basemap(ax)
    
#     _draw_scale_bar(ax, extent, loc="lower right", length_m=1000)
#     ax.set_axis_off()

#     if save:
#         out = os.path.join(MAP_DIR, f"{SAFE(location_name)}_multimode_{int(travel_time_minutes)}min.png")
#         plt.tight_layout()
#         plt.savefig(out, bbox_inches="tight", dpi=200)
#         print(f"[OK] Saved map: {out}")
#     plt.close(fig)

# -------------------------------
# Plot: single-mode frontiers
# -------------------------------
def plot_single_mode_frontiers(
    location_name: str,
    mode: str,
    pois: List[POI],                 # real names + coords
    poi_polys: List[gpd.GeoSeries],  # one GeoSeries per POI, EPSG:4326
    base_graphs: List[nx.MultiDiGraph],
    travel_time_minutes: float,
    fade_steps: int = 6,
    fade_max: float = 0.25,
    save: bool = True
):
    # ---------- publication settings ----------
    FIGSIZE_IN  = (11.7, 8.3)   # A4 landscape
    OUTPUT_DPI  = 300           # print-quality
    MARGIN_PAD  = 0.04          # ~4% padding around extent
    # -----------------------------------------

    extent = _graphs_extent_3857(base_graphs)
    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=OUTPUT_DPI)

    # Faint base networks
    for G in base_graphs:
        edges = _edges_gdf_3857(G)  # returns 3414
        if not edges.empty:
            edges.plot(ax=ax, lw=0.6, color="#BDBDBD", alpha=0.6)

    style = MODE_STYLE.get(mode, {"color": "#777", "label": mode.title()})
    color = style["color"]

    # Frontiers with fade effect
    for gseries in poi_polys:
        if gseries is None or len(gseries) == 0:
            continue
        crs_in = getattr(gseries, "crs", None) or 4326
        gdf3414 = gpd.GeoDataFrame(geometry=gseries).set_crs(crs_in, allow_override=True).to_crs(3414)
        geom = gdf3414.geometry.iloc[0]

        polys: List[Polygon] = []
        if isinstance(geom, Polygon):
            polys = [geom]
        elif isinstance(geom, MultiPolygon):
            polys = list(geom.geoms)

        for poly in polys:
            for i in range(fade_steps, 0, -1):
                alpha = fade_max * (i / fade_steps)
                _draw_polygon(ax, poly, facecolor=color, edgecolor=color, alpha_fill=alpha, lw=0.9)
            _draw_polygon(ax, poly, facecolor="none", edgecolor=color, alpha_fill=0.0, lw=2.0)

    # POIs and labels
    _plot_pois_with_labels(ax, pois)

    ax.set_title(f"{location_name}: {style['label']} {_format_duration(travel_time_minutes)} Frontier", **TITLE_FONTS)
    ax.text(0.01, 0.98, "Reachable area by travel time • OSM network • Tiles © CartoDB",
            transform=ax.transAxes, ha="left", va="top", **SUBTITLE_FONTS)

    # Legend
    handles = [
        Patch(facecolor=color, edgecolor=color, alpha=0.55, label=f"{style['label']} frontier"),
        Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor="#111",
               markersize=8, label="POI"),
    ]
    ax.legend(handles=handles, frameon=True, framealpha=0.92, loc="upper right")

    # Uniform padding + fixed aspect
    def _expand_extent(ext, pad=MARGIN_PAD):
        minx, maxx, miny, maxy = ext
        dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
        return (minx - dx, maxx + dx, miny - dy, maxy + dy)

    extent = _expand_extent(extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    # Basemap AFTER limits so tiles match view
    _add_basemap(ax)

    _draw_scale_bar(ax, extent, loc="lower right", length_m=1000)
    ax.set_axis_off()

    if save:
        out = os.path.join(MAP_DIR, f"{SAFE(location_name)}_{mode}_{int(travel_time_minutes)}min.png")
        plt.savefig(out, bbox_inches="tight", dpi=OUTPUT_DPI,
                    facecolor=fig.get_facecolor(), transparent=False)
        print(f"[OK] Saved map: {out} ({FIGSIZE_IN[0]}x{FIGSIZE_IN[1]} in @ {OUTPUT_DPI} dpi)")
    plt.close(fig)


def plot_multi_mode_overlay(
    location_name: str,
    pois: List[POI],
    per_mode_polys: Dict[str, List[gpd.GeoSeries]],   # mode -> list of GeoSeries
    base_graphs_all: List[nx.MultiDiGraph],
    travel_time_minutes: float,
    save: bool = True
):
    # ---------- publication settings ----------
    FIGSIZE_IN  = (11.7, 8.3)   # A4 landscape
    OUTPUT_DPI  = 300
    MARGIN_PAD  = 0.04
    # -----------------------------------------

    extent = _graphs_extent_3857(base_graphs_all)
    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=OUTPUT_DPI)

    # Base networks (light)
    for G in base_graphs_all:
        edges = _edges_gdf_3857(G)  # returns 3414
        if not edges.empty:
            edges.plot(ax=ax, lw=0.5, color="#BDBDBD", alpha=0.6)

    # ------- Per-mode polygons with explicit bottom→top order -------
    draw_order = ["drive", "bike", "walk"]
    draw_order += [m for m in per_mode_polys.keys() if m not in draw_order]

    handles = []
    for mode in draw_order:
        if mode not in per_mode_polys:
            continue
        style = MODE_STYLE.get(mode, {"color": "#777", "label": mode.title()})
        color = style["color"]
        any_drawn = False

        for gseries in per_mode_polys[mode]:
            if gseries is None or len(gseries) == 0:
                continue
            crs_in = getattr(gseries, "crs", None) or 4326
            gdf3414 = gpd.GeoDataFrame(geometry=gseries).set_crs(crs_in, allow_override=True).to_crs(3414)
            geom = gdf3414.geometry.iloc[0]

            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            else:
                polys = []

            for poly in polys:
                _draw_polygon(ax, poly, facecolor=color, edgecolor=color, alpha_fill=0.35, lw=2.0)
                any_drawn = True

        if any_drawn:
            handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=style["label"]))
    # ----------------------------------------------------------------

    # POIs and labels (+legend handle)
    _plot_pois_with_labels(ax, pois)
    handles.append(Line2D([0], [0], marker="o", lw=0, markerfacecolor="white", markeredgecolor="#111",
                          markersize=8, label="POI"))

    ax.set_title(f"{location_name}: Multi-mode {_format_duration(travel_time_minutes)} Frontiers", **TITLE_FONTS)
    ax.text(0.01, 0.98, "Reachable areas by mode and travel time • OSM network • Tiles © CartoDB",
            transform=ax.transAxes, ha="left", va="top", **SUBTITLE_FONTS)

    if handles:
        ax.legend(handles=handles, frameon=True, framealpha=0.92, loc="upper right")

    # Uniform padding + fixed aspect
    def _expand_extent(ext, pad=MARGIN_PAD):
        minx, maxx, miny, maxy = ext
        dx, dy = (maxx - minx) * pad, (maxy - miny) * pad
        return (minx - dx, maxx + dx, miny - dy, maxy + dy)

    extent = _expand_extent(extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

    # Basemap AFTER limits so tiles match the view
    _add_basemap(ax)

    _draw_scale_bar(ax, extent, loc="lower right", length_m=1000)
    ax.set_axis_off()

    if save:
        out = os.path.join(MAP_DIR, f"{SAFE(location_name)}_multimode_{int(travel_time_minutes)}min.png")
        plt.savefig(out, bbox_inches="tight", dpi=OUTPUT_DPI,
                    facecolor=fig.get_facecolor(), transparent=False)
        print(f"[OK] Saved map: {out} ({FIGSIZE_IN[0]}x{FIGSIZE_IN[1]} in @ {OUTPUT_DPI} dpi)")
    plt.close(fig)


# -------------------------------
# Pipeline
# -------------------------------
def run_pipeline(
    location_name: str,
    poi_inputs: Iterable,                 # List[str] or List[POI]
    modes: Iterable[str] = ("walk", "bike", "drive"),
    durations_min: Iterable[float] = (15,),
    speeds_kmh: Optional[Dict[str, float]] = None,
    node_penalty_min: Optional[float] = None,
    save_figs: bool = True,
    export_vectors: bool = False
) -> None:
    """
    Orchestrates:
      1) Resolve POIs
      2) Download graphs per POI+mode
      3) Precompute nearest nodes
      4) For each duration and mode: compute reachable subgraphs and boundary polygons
      5) Plot per-mode and multi-mode maps
      6) Optionally export vector boundaries
    """
    if speeds_kmh is None:
        speeds_kmh = MODE_SPEEDS_KMH
    if node_penalty_min is None:
        node_penalty_min = NODE_PENALTY_MIN

    # Resolve POIs
    pois = resolve_pois(poi_inputs)
    if not pois:
        print("[ERROR] No valid POIs after geocoding.")
        return
    print(f"[INFO] Using {len(pois)} POIs: {[p.name for p in pois]}")

    # Download graphs per mode+poi
    per_mode_graphs: Dict[str, List[nx.MultiDiGraph]] = {m: [] for m in modes}
    for mode in modes:
        for poi in pois:
            G = download_graph(poi.lat, poi.lon, mode)
            per_mode_graphs[mode].append(G)

    all_graphs = [G for m in modes for G in per_mode_graphs[m]]

    # Nearest nodes for routing
    nearest_nodes: Dict[str, List[int]] = {m: [] for m in modes}
    for mode in modes:
        for poi, G in zip(pois, per_mode_graphs[mode]):
            node = nearest_node_no_sklearn(G, poi.lon, poi.lat)
            nearest_nodes[mode].append(node)

    # For each duration compute and plot
    for dur in durations_min:
        print(f"[INFO] Computing frontiers for duration={dur} minutes")
        per_mode_polys: Dict[str, List[gpd.GeoSeries]] = {m: [] for m in modes}

        for mode in modes:
            speed = speeds_kmh.get(mode, 5.0)
            mode_graphs = per_mode_graphs[mode]
            mode_nodes = nearest_nodes[mode]
            mode_polys: List[gpd.GeoSeries] = []

            for poi, G, center_node in zip(pois, mode_graphs, mode_nodes):
                subG = get_reachable_subgraph(
                    G, center_node=center_node,
                    travel_time_minutes=dur,
                    speed_kmh=speed,
                    node_penalty_min=node_penalty_min
                )
                gseries = create_outer_boundary(subG, buffer_extra_m=10.0)
                gseries.name = poi.name
                mode_polys.append(gseries)

                if export_vectors and (gseries is not None) and (len(gseries) > 0):
                    base_name = f"{location_name}_{mode}_{poi.name}_{int(dur)}min"
                    export_boundary(gseries, base_name)

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

        if save_figs and len(modes) > 1:
            plot_multi_mode_overlay(
                location_name=location_name,
                pois=pois,
                per_mode_polys=per_mode_polys,
                base_graphs_all=all_graphs,
                travel_time_minutes=dur,
                save=True
            )

    print("[OK] Pipeline complete.")

# -------------------------------
# Example entrypoint
# -------------------------------
if __name__ == "__main__":
    LOCATION = "test_sg_2"
    POI_NAMES = ["NUS University Town", "Singapore Botanic Gardens", "HortPark"]
    MODES = ["walk","bike","drive"] 
    DURATIONS = [13]

    run_pipeline(
        location_name=LOCATION,
        poi_inputs=POI_NAMES,
        modes=MODES,
        durations_min=DURATIONS,
        save_figs=True,
        export_vectors=True,
    )