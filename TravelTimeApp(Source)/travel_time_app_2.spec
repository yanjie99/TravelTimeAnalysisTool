# travel_time_app.spec
# Build:  pyinstaller --clean travel_time_app.spec
# Output: dist/TravelTimeApp(.exe)

from PyInstaller.utils.hooks import collect_all, copy_metadata
from PyInstaller.building.build_main import Analysis, PYZ, EXE

# --- Collect everything needed from GIS stack (code, data, C-extensions, metadata)
datas, binaries, hiddenimports = [], [], []

for pkg in [
    "osmnx",
    "contextily",
    "rasterio",
    "fiona",
    "shapely",
    "pyproj",
    "xyzservices",   # used by contextily for tile providers
]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Ensure importlib.metadata.version(...) calls don’t fail
for pkg in ["osmnx", "contextily", "rasterio", "xyzservices"]:
    datas += copy_metadata(pkg)

# Pillow plugins and Matplotlib Agg backend (headless)
hiddenimports += [
    "matplotlib.backends.backend_agg",
    "PIL.Image",
    "PIL.ImageFile",
    "PIL.PngImagePlugin",
    "PIL.JpegImagePlugin",
    "PIL.GifImagePlugin",
    "PIL.TiffImagePlugin",
    "certifi",   # useful for HTTPS tile servers on some Windows setups
]

a = Analysis(
    ["UI_TraTA_app4.py"],            # entrypoint
    pathex=[],                       # you can add extra search paths if needed
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],                    # custom hook dirs (not needed here)
    runtime_hooks=["hook_set_gis_env.py"],   # sets PROJ/GDAL env vars at runtime
    excludes=[],                     # add exclusions if you want to shrink
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="TravelTimeApp",
    console=False,         # set True if you want a console for logs
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,              # set False if UPX not installed
    upx_exclude=[],
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,             # e.g., "app.ico" on Windows, ".icns" on macOS
)
