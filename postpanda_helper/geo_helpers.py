# coding=utf-8
from collections import MutableMapping
from typing import Optional

from pandas import DataFrame, Series
from sqlalchemy import Table
from sqlalchemy.sql.type_api import UserDefinedType

HAS_GEO_EXTENSIONS = False

try:
    from ._geo_helpers import (
        _df_to_shape,
        _fill_geoseries,
        get_geometry_type,
        geometry_to_ewkb,
        _convert_geometry_for_postgis,
    )

    df_to_shape = _df_to_shape
    fill_geoseries = _fill_geoseries
    convert_geometry_for_postgis = _convert_geometry_for_postgis
    HAS_GEO_EXTENSIONS = True
except ImportError:

    def df_to_shape(tbl: Table, frame: DataFrame) -> None:
        pass

    def fill_geoseries(s: Series) -> (Series, bool):
        return s, False

    def convert_geometry_for_postgis(
        frame: DataFrame, column: str, in_place: bool = False
    ) -> (Optional[DataFrame], MutableMapping[str, UserDefinedType]):
        return None, {}
