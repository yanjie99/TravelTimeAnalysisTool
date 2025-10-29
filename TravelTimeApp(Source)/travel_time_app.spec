
# travel_time_app.spec
# Build with:  pyinstaller travel_time_app.spec
# Produces:    dist/TravelTimeApp  (or TravelTimeApp.exe on Windows)

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
from PyInstaller.building.build_main import Analysis, PYZ, EXE
import os

block_cipher = None

# Collect data/binaries that GIS stack commonly needs
datas = []
binaries = []

# PROJ data (pyproj), GDAL-related (via fiona), shapely data
datas += collect_data_files('pyproj')
datas += collect_data_files('fiona')
datas += collect_data_files('shapely')

# NEW: include OSMnx package metadata (required for importlib.metadata.version)
datas += copy_metadata('osmnx')

# (optional but safe) include osmnx data files too
datas += collect_data_files('osmnx')

# Native libs/DLLs: shapely, rtree
binaries += collect_dynamic_libs('shapely')
binaries += collect_dynamic_libs('rtree')

hiddenimports = [
    # matplotlib headless backend
    'matplotlib.backends.backend_agg',
    # pillow plugins (avoid "cannot identify image file" at runtime)
    'PIL.Image',
    'PIL.ImageFile',
    'PIL.PngImagePlugin',
    'PIL.JpegImagePlugin',
    'PIL.GifImagePlugin',
    'PIL.TiffImagePlugin',
]

a = Analysis(
    ['UI_TraTA_app4.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=['hook_set_gis_env.py'],  # sets PROJ_LIB etc. at runtime
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TravelTimeApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,    # set True if you want a console window for logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
