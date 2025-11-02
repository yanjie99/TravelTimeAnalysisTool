
import time, geopandas as gpd, osmnx as ox
from .models import POI

def geocode_freeform(q: str, retries=3, pause=1.0):
    for i in range(retries):
        try:
            res = ox.geocoder.geocode(q)
            if isinstance(res, (list, tuple)) and len(res) == 2:
                lat, lon = float(res[0]), float(res[1])
            else:
                geom = gpd.GeoSeries([res], crs=4326)
                lon, lat = geom.to_crs(4326).geometry.iloc[0].centroid.coords[0]
            print(f"[OK] Geocoded '{q}' -> ({lat:.6f}, {lon:.6f})")
            return POI(name=q, lat=lat, lon=lon)
        except Exception as e:
            print(f"[WARN] Geocode attempt {i+1} failed for '{q}': {e}")
            time.sleep(pause * (2**i))
    print(f"[ERROR] Failed to geocode '{q}'")
    return None

def resolve_pois(poi_inputs):
    out = []
    for item in poi_inputs:
        if hasattr(item, "lat") and hasattr(item, "lon"):
            out.append(item)
        elif isinstance(item, str):
            p = geocode_freeform(item)
            if p: out.append(p)
        else:
            print(f"[WARN] Unsupported POI input: {item!r}")
    return out
