# -*- coding: utf-8 -*-
from typing import MutableMapping, Optional, Tuple

from pandas import DataFrame, Series
from sqlalchemy import Table
from sqlalchemy.sql.type_api import UserDefinedType

HAS_GEO_EXTENSIONS = False

try:
    from ._geo_helpers import (  # noqa: F401
        _convert_geometry_for_postgis,
        _df_to_shape,
        _fill_geoseries,
        geometry_to_ewkb,
        get_geometry_type,
    )

    df_to_shape = _df_to_shape
    fill_geoseries = _fill_geoseries
    convert_geometry_for_postgis = _convert_geometry_for_postgis
    HAS_GEO_EXTENSIONS = True
except ImportError:

    def df_to_shape(tbl: Table, frame: DataFrame) -> None:
        pass

    def fill_geoseries(s: Series) -> Tuple[Series, bool]:
        return s, False

    def convert_geometry_for_postgis(
        frame: DataFrame, column: str, in_place: bool = False
    ) -> Tuple[Optional[DataFrame], MutableMapping[str, UserDefinedType]]:
        return None, {}
