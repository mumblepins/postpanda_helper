# coding=utf-8

import numpy as np
import pandas as pd
from psycopg2.extensions import AsIs, adapt, register_adapter
from sqlalchemy.dialects.postgresql import DATERANGE, NUMRANGE, TSRANGE, TSTZRANGE

from .geo_helpers import HAS_GEO_EXTENSIONS
from .pd_helpers import is_date_interval, interval_to_range
from .psql_helpers import PandaCSVtoSQL
from .select_sert import SelectSert

if HAS_GEO_EXTENSIONS:
    from .geo_helpers import get_geometry_type, geometry_to_ewkb
    from geopandas.array import GeometryDtype
    from geopandas import GeoSeries
    from geoalchemy2.types import Geometry

register_adapter(type(pd.NA), lambda x: adapt(None))
register_adapter(np.int64, lambda x: AsIs(x))
register_adapter(pd.Interval, interval_to_range)


def pd_to_sql(
    frame,
    name,
    con,
    schema=None,
    if_exists="fail",
    index=True,
    index_label=None,
    chunksize=None,
    dtype=None,
    method=None,
):
    """
    With support for daterange and tsrange

    """
    dtype = dtype or {}
    copied = False
    columns = frame.columns
    for c in columns:
        if c in dtype:
            continue
        if pd.api.types.is_interval_dtype(frame[c]):
            ex = frame[c][frame[c].first_valid_index()]
            if isinstance(ex.left, pd.Timestamp):
                if is_date_interval(ex):
                    dtype[c] = DATERANGE
                elif hasattr(ex, "tz") and ex.tz is not None:
                    dtype[c] = TSTZRANGE
                else:
                    dtype[c] = TSRANGE
            elif pd.api.types.is_number(ex.left):
                dtype[c] = NUMRANGE
        elif HAS_GEO_EXTENSIONS and isinstance(frame[c].dtype, GeometryDtype):
            # copy because we're going to mess with the column
            if not copied:
                frame = frame.copy()
            s = GeoSeries(frame[c])
            try:
                geometry_type, has_curve = get_geometry_type(s)
            except ValueError:
                frame[c] = frame[c].astype("object")
                continue
            srid = s.crs.to_epsg(min_confidence=25) or -1
            dtype[c] = Geometry(geometry_type=geometry_type, srid=srid)
            frame[c] = geometry_to_ewkb(frame[c], srid)

    dtype = dtype or None
    return frame.to_sql(name, con, schema, if_exists, index, index_label, chunksize, dtype, method)


__all__ = ["PandaCSVtoSQL", "SelectSert", "pd_to_sql"]
