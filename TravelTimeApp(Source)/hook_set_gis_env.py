
# hook_set_gis_env.py
# Runtime hook to make packaged app find PROJ/GEOS/GDAL data at runtime.

import os

# Ensure PROJ data (used by pyproj/PROJ) can be found
try:
    from pyproj.datadir import get_data_dir
    os.environ.setdefault('PROJ_LIB', get_data_dir())
except Exception:
    pass

# Fiona/GDAL tends to init its own environment; importing ensures GDAL data is registered
try:
    import fiona  # noqa: F401
except Exception:
    pass

# Shapely GEOS location is collected via collect_dynamic_libs in the spec
# Rtree (spatialindex) is also collected similarly.
