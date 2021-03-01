# coding=utf-8

import numpy as np
import pandas as pd
from psycopg2.extensions import AsIs, adapt, register_adapter
from sqlalchemy.dialects.postgresql import DATERANGE, NUMRANGE, TSRANGE, TSTZRANGE

from .pd_helpers import is_date_interval, interval_to_range
from .psql_helpers import PandaCSVtoSQL
from .select_sert import SelectSert

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
    for c in frame.columns:
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
    dtype = dtype or None
    return frame.to_sql(name, con, schema, if_exists, index, index_label, chunksize, dtype, method)


__all__ = ["PandaCSVtoSQL", "SelectSert", "pd_to_sql"]
