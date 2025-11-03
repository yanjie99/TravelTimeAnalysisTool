import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

def SAFE(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s).strip("_")

def _expand_extent(ext, pad_ratio=0.04):
    minx, maxx, miny, maxy = ext
    dx, dy = (maxx - minx) * pad_ratio, (maxy - miny) * pad_ratio
    return (minx - dx, maxx + dx, miny - dy, maxy + dy)

def format_duration(tt: float) -> str:
    return f"{int(tt)} min" if float(tt).is_integer() else f"{tt:.1f} min"

def polys_extent_3414(series_lists, pad_ratio=0.06):
    polys_3414 = []
    for gs in series_lists:
        if gs is None or len(gs) == 0:
            continue
        crs_in = getattr(gs, "crs", None) or 4326
        gdf = gpd.GeoDataFrame(geometry=gs).set_crs(crs_in, allow_override=True).to_crs(3414)
        geom = gdf.geometry.iloc[0]
        if not geom.is_empty:
            polys_3414.append(geom.envelope)
    if polys_3414:
        union = unary_union(polys_3414)
        minx, miny, maxx, maxy = union.bounds
    else:
        minx, miny, maxx, maxy = gpd.GeoSeries([Point(103.8198, 1.3521)], crs=4326).to_crs(3414).buffer(15000).total_bounds
    return _expand_extent((minx, maxx, miny, maxy), pad_ratio)

def graphs_extent_3414(graphs, edges_gdf_3414_fn):
    polys = []
    for G in graphs:
        edges = edges_gdf_3414_fn(G)
        if not edges.empty:
            polys.append(edges.unary_union.envelope)
    if polys:
        union = unary_union(polys)
        minx, miny, maxx, maxy = union.bounds
    else:
        minx, miny, maxx, maxy = gpd.GeoSeries([Point(103.8198, 1.3521)], crs=4326).to_crs(3414).buffer(15000).total_bounds
    return _expand_extent((minx, maxx, miny, maxy), 0.06)
