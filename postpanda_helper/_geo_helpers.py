import geopandas as gpd
from geoalchemy2 import Geometry
from geoalchemy2.shape import to_shape
from geopandas import GeoSeries
from pandas import DataFrame, Series
from shapely.geometry import Point
from shapely.geos import lgeos
from shapely.wkb import dumps
from sqlalchemy import Table

_NA_FILL_GEOM = Point(3.141, 59.265)


def _df_to_shape(tbl: Table, frame: DataFrame) -> None:
    geo_cols = [c.name for c in tbl.c if isinstance(c.type, Geometry)]
    for gc in geo_cols:
        notna = frame[gc].notna()
        if notna.sum() == 0:
            continue
        frame.loc[notna, gc] = frame.loc[notna, gc].apply(to_shape)

        srid = lgeos.GEOSGetSRID(frame.loc[notna, gc].iloc[0]._geom)
        # if no defined SRID in geodatabase, returns SRID of 0
        crs = None
        if srid != 0:
            crs = "epsg:{}".format(srid)
        frame[gc] = gpd.GeoSeries(frame[gc], crs=crs)


def _fill_geoseries(s: Series) -> (Series, bool):
    gs = False
    if isinstance(s.dtype, gpd.array.GeometryDtype):
        s = s.fillna(_NA_FILL_GEOM)
        gs = True
    return s, gs


def get_geometry_type(gs: "GeoSeries"):
    geom_types = list(gs.geom_type.unique())
    has_curve = False

    for gt in geom_types:
        if gt is None:
            continue
        elif "LinearRing" in gt:
            has_curve = True

    if len(geom_types) == 1:
        if has_curve:
            target_geom_type = "LINESTRING"
        else:
            if geom_types[0] is None:
                raise ValueError("No valid geometries in the data.")
            else:
                target_geom_type = geom_types[0].upper()
    else:
        target_geom_type = "GEOMETRY"

    # Check for 3D-coordinates
    if any(gs.has_z):
        target_geom_type = target_geom_type + "Z"

    return target_geom_type, has_curve


def geometry_to_ewkb(gs: GeoSeries, srid):
    return gs.apply(lambda x: dumps(x, srid=srid, hex=True))
