
import os

# App-rooted output paths (fixes “Open Maps Folder” mismatch in GUIs)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MAP_DIR  = os.path.join(APP_ROOT, "maps")
VEC_DIR  = os.path.join(APP_ROOT, "frontiers")
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(VEC_DIR, exist_ok=True)

MODE_SPEEDS_KMH = {"walk": 5.0, "bike": 15.0, "drive": 35.0}
SEARCH_RADIUS_M = {"walk": 1500, "bike": 6250, "drive": 10000}
NODE_PENALTY_MIN = 0.02

MODE_STYLE = {
    "walk":  {"color": "#007635", "label": "Walk"},
    "bike":  {"color": "#F87C00", "label": "Bike"},
    "drive": {"color": "#7636E3", "label": "Drive"},
}

TITLE_FONTS = dict(fontsize=14, fontweight="bold")
SUBTITLE_FONTS = dict(fontsize=10, color="#555")
