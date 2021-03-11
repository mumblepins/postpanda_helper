# coding=utf-8

from numpy import int64 as _int64
from pandas import NA as _NA, Interval as _Interval
from psycopg2.extensions import (
    AsIs as _AsIs,
    adapt as _adapt,
    register_adapter as _register_adapter,
)

from .pandas_reader import PandasReader
from .pandas_writer import PandasWriter
from .pd_helpers import interval_to_range as _interval_to_range
from .psql_helpers import PandaCSVtoSQL
from .select_sert import SelectSert

_register_adapter(type(_NA), lambda x: _adapt(None))
_register_adapter(_int64, lambda x: _AsIs(x))
_register_adapter(_Interval, _interval_to_range)

__all__ = ["PandaCSVtoSQL", "SelectSert", "PandasReader", "PandasWriter"]
