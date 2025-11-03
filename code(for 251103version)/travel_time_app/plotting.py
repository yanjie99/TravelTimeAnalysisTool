import matplotlib
matplotlib.use("Agg", force=True)

import os, numpy as np, geopandas as gpd, matplotlib.pyplot as plt, contextily as cx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Polygon, MultiPolygon, Point
from .config import MAP_DIR, MODE_STYLE, TITLE_FONTS
from .utils import format_duration, graphs_extent_3414, polys_extent_3414
from .graphs import _edges_gdf_3414

def _add_basemap(ax, zoom="auto"):
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron,
                       crs="EPSG:3414", zoom=zoom, reset_extent=False, attribution=False)
    except Exception as e:
        print(f"[WARN] Basemap failed: {e}")

def _draw_scale_bar(ax, extent_3414, loc="lower right", length_m=1000):
    minx, maxx, miny, maxy = extent_3414
    pad_x = (maxx - minx) * 0.04; pad_y = (maxy - miny) * 0.04
    y = miny + pad_y if "lower" in loc else maxy - pad_y
    x_right = maxx - pad_x if "right" in loc else minx + pad_x
    x0 = x_right - length_m
    ax.plot([x0, x_right], [y, y], lw=3, solid_capstyle="butt", color="#000000")
    ax.text((x0 + x_right)/2, y + pad_y*0.25,
            f"{length_m/1000:.0f} km" if length_m >= 1000 else f"{length_m:.0f} m",
            ha="center", va="bottom", fontsize=9, color="#222",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=2))

def _plot_pois_with_symbols(ax, pois):
    if not pois: return []
    poi_gdf = gpd.GeoDataFrame(
        {"name": [p.name for p in pois]},
        geometry=[Point(p.lon, p.lat) for p in pois],
        crs=4326
    ).to_crs(3414)

    markers = ["o", "s", "^", "D", "P"]
    colors  = ["#1E5DE4", "#FC8D62", "#66C2A5", "#8DA0CB", "#E78AC3"]

    handles = []
    for i, (_, r) in enumerate(poi_gdf.iterrows()):
        m = markers[i % len(markers)]
        c = colors[i % len(colors)]
        ax.scatter(r.geometry.x, r.geometry.y, s=40, c=c, edgecolor="#111", lw=0.9, marker=m, zorder=8)
        handles.append(Line2D([0],[0], marker=m, color="none",
                              markerfacecolor=c, markeredgecolor="#111", markersize=8, label=r["name"]))
    return handles

def _draw_polygon(ax, poly: Polygon, facecolor: str, edgecolor: str, alpha_fill: float, lw: float = 1.4):
    coords = np.asarray(poly.exterior.coords)
    ax.add_patch(plt.Polygon(coords, fill=True, facecolor=facecolor, alpha=alpha_fill, edgecolor=edgecolor, lw=lw))

def _legend_frontier_patch(color: str, label: str) -> Patch:
    return Patch(facecolor=color, edgecolor=color, alpha=0.25, label=label)

def plot_single_mode_frontiers(location_name, mode, pois, poi_polys, base_graphs, travel_time_minutes,
                               fade_steps=6, fade_max=0.25, save=True, show_network=True):
    FIGSIZE_IN, OUTPUT_DPI = (8.3, 11.7), 300
    extent = (polys_extent_3414(poi_polys) if not show_network
              else graphs_extent_3414(base_graphs, _edges_gdf_3414))
    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=OUTPUT_DPI)

    if show_network:
        for G in base_graphs:
            edges = _edges_gdf_3414(G)
            if not edges.empty:
                edges.plot(ax=ax, lw=0.6, color="#10D4DB", alpha=0.1)

    style = MODE_STYLE.get(mode, {"color": "#777", "label": mode.title()})
    color = style["color"]

    for gseries in poi_polys:
        if gseries is None or len(gseries) == 0: continue
        crs_in = getattr(gseries, "crs", None) or 4326
        gdf3414 = gpd.GeoDataFrame(geometry=gseries).set_crs(crs_in, allow_override=True).to_crs(3414)
        geom = gdf3414.geometry.iloc[0]
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
        for poly in polys:
            for i in range(fade_steps, 0, -1):
                alpha = fade_max * (i / fade_steps)
                _draw_polygon(ax, poly, facecolor=color, edgecolor=color, alpha_fill=alpha, lw=0.9)
            _draw_polygon(ax, poly, facecolor="none", edgecolor=color, alpha_fill=0, lw=2.0)

    handles = [_legend_frontier_patch(color, f"{style['label']} frontier")]
    handles += _plot_pois_with_symbols(ax, pois)

    ax.set_title(f"{location_name}: {style['label']} {format_duration(travel_time_minutes)} Frontier", **TITLE_FONTS)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    _add_basemap(ax)
    _draw_scale_bar(ax, extent, loc="upper right", length_m=1000)
    ax.set_axis_off()
    if handles:
        ax.legend(handles=handles, frameon=True, framealpha=0.92, loc="lower right")
    if save:
        out = os.path.join(MAP_DIR, f"{location_name}_{mode}_{int(travel_time_minutes)}min.png".replace(" ", "_"))
        plt.savefig(out, bbox_inches="tight", dpi=OUTPUT_DPI, facecolor=fig.get_facecolor(), transparent=False)
        print(f"[OK] Saved map: {out} ({FIGSIZE_IN[0]}x{FIGSIZE_IN[1]} in @ {OUTPUT_DPI} dpi)")
    plt.close(fig)

def plot_multi_mode_overlay(location_name, pois, per_mode_polys, base_graphs_all, travel_time_minutes,
                            save=True, show_network=False):
    FIGSIZE_IN, OUTPUT_DPI = (8.3, 11.7), 300
    all_series = [gs for lst in per_mode_polys.values() for gs in lst]
    extent = (polys_extent_3414(all_series) if not show_network
              else graphs_extent_3414(base_graphs_all, _edges_gdf_3414))
    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=OUTPUT_DPI)

    if show_network:
        for G in base_graphs_all:
            edges = _edges_gdf_3414(G)
            if not edges.empty:
                edges.plot(ax=ax, lw=0.5, color="#BDBDBD", alpha=0.6)

    draw_order = ["drive", "bike", "walk"] + [m for m in per_mode_polys.keys() if m not in ("drive","bike","walk")]
    handles = []
    for mode in draw_order:
        if mode not in per_mode_polys: continue
        style = MODE_STYLE.get(mode, {"color": "#777", "label": mode.title()})
        color = style["color"]; any_drawn = False
        for gseries in per_mode_polys[mode]:
            if gseries is None or len(gseries) == 0: continue
            crs_in = getattr(gseries, "crs", None) or 4326
            gdf3414 = gpd.GeoDataFrame(geometry=gseries).set_crs(crs_in, allow_override=True).to_crs(3414)
            geom = gdf3414.geometry.iloc[0]
            polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms) if isinstance(geom, MultiPolygon) else []
            for poly in polys:
                _draw_polygon(ax, poly, facecolor=color, edgecolor=color, alpha_fill=0.25, lw=2.0)
                any_drawn = True
        if any_drawn:
            handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=style["label"]))

    handles += _plot_pois_with_symbols(ax, pois)
    ax.set_title(f"{location_name}: Multi-mode {format_duration(travel_time_minutes)} Frontiers", **TITLE_FONTS)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    _add_basemap(ax)
    _draw_scale_bar(ax, extent, loc="upper right", length_m=1000)
    ax.set_axis_off()
    if handles:
        ax.legend(handles=handles, frameon=True, framealpha=0.92, loc="lower right")
    if save:
        out = os.path.join(MAP_DIR, f"{location_name}_multimode_{int(travel_time_minutes)}min.png".replace(" ", "_"))
        plt.savefig(out, bbox_inches="tight", dpi=OUTPUT_DPI, facecolor=fig.get_facecolor(), transparent=False)
        print(f"[OK] Saved map: {out} ({FIGSIZE_IN[0]}x{FIGSIZE_IN[1]} in @ {OUTPUT_DPI} dpi)")
    plt.close(fig)
