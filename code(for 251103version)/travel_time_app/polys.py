import os, geopandas as gpd, numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from .utils import SAFE
from .config import VEC_DIR

def create_outer_boundary(reachable_subgraph, buffer_extra_m: float = 10.0) -> gpd.GeoSeries:
    empty_3414 = gpd.GeoSeries([], dtype=object, crs=3414)
    if reachable_subgraph is None or reachable_subgraph.number_of_edges() == 0:
        return empty_3414
    import travel_time_app.graphs as graphs
    edges3414 = graphs._edges_gdf_3414(reachable_subgraph)
    if edges3414.empty: return empty_3414

    lengths = edges3414.length.values
    mean_len = float(np.nanmean(lengths)) if len(lengths) else 20.0
    base = max(5.0, min(60.0, mean_len * 0.3))
    buf_dist = base + float(buffer_extra_m)

    lines = unary_union(edges3414.geometry.values)
    if lines.is_empty: return empty_3414

    buffered = lines.buffer(buf_dist, cap_style=2, join_style=2)

    if isinstance(buffered, Polygon):
        poly = buffered
    elif isinstance(buffered, MultiPolygon):
        if len(buffered.geoms) == 0: return empty_3414
        poly = max(buffered.geoms, key=lambda p: p.area)
    else:
        geoms = [g for g in getattr(buffered, "geoms", []) if isinstance(g, Polygon)]
        if not geoms: return empty_3414
        poly = max(geoms, key=lambda p: p.area)

    return gpd.GeoSeries([poly], crs=3414)

def export_boundary(gseries: gpd.GeoSeries, base_name: str) -> None:
    if gseries is None or len(gseries) == 0:
        print(f"[WARN] Nothing to export for {base_name}")
        return
    gs = gseries
    crs_in = getattr(gs, "crs", None)
    if crs_in is None:
        try:
            minx, miny, maxx, maxy = gpd.GeoSeries(gs, dtype=object).total_bounds
            projected_like = (max(abs(minx), abs(maxx)) > 500) or (max(abs(miny), abs(maxy)) > 500)
            assumed_crs = 3414 if projected_like else 4326
            gs = gpd.GeoSeries(gs, crs=assumed_crs)
        except Exception:
            gs = gpd.GeoSeries(gs, crs=4326)
    gs_4326 = gs.to_crs(4326) if gs.crs != 4326 else gs

    safe = SAFE(base_name)
    os.makedirs(VEC_DIR, exist_ok=True)
    gdf = gpd.GeoDataFrame({"name": [base_name]}, geometry=gs_4326, crs=4326)
    gdf.to_file(os.path.join(VEC_DIR, f"{safe}.geojson"), driver="GeoJSON")
    gdf.to_file(os.path.join(VEC_DIR, f"{safe}.gpkg"),   driver="GPKG")
    print(f"[OK] Exported: {os.path.join(VEC_DIR, f'{safe}.geojson')}  |  {os.path.join(VEC_DIR, f'{safe}.gpkg')}")
