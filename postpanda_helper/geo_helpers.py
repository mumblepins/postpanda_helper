# coding=utf-8
from pandas import DataFrame, Series
from sqlalchemy import Table

HAS_GEO_EXTENSIONS = False

try:
    from ._geo_helpers import _df_to_shape, _fill_geoseries, get_geometry_type, geometry_to_ewkb

    df_to_shape = _df_to_shape
    fill_geoseries = _fill_geoseries
    HAS_GEO_EXTENSIONS = True
except ImportError:

    def df_to_shape(tbl: Table, frame: DataFrame) -> None:
        pass

    def fill_geoseries(s: Series) -> (Series, bool):
        return s, False
