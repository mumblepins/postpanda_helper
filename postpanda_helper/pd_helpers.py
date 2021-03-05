# coding=utf-8
import random
import string
from contextlib import contextmanager
from datetime import date
from typing import Union

import numpy as np
import pandas as pd
from psycopg2._psycopg import adapt
from psycopg2._range import DateRange, DateTimeTZRange, DateTimeRange, NumericRange

from postpanda_helper.geo_helpers import fill_geoseries

_NA_FILL_INTEGER = np.uint64(random.SystemRandom().getrandbits(64)).astype(np.int64)
_NA_FILL_OBJECT = "na_val_" + "".join(random.SystemRandom().choices(string.ascii_lowercase, k=8))

DATE_FORMATS = {4: "%Y", 6: "%Y%m", 8: "%Y%m%d"}


def downcast(data: pd.Series) -> pd.Series:
    """
    Downcasts integer types to smallest possible type
    Args:
        data:

    Returns:

    """
    if not pd.api.types.is_integer_dtype(data):
        return data
    dmax = data.fillna(0).max()
    for n in range(3, 7):
        nmax = np.iinfo(f"int{2 ** n}").max
        if dmax <= nmax:
            return data.astype(f"Int{2 ** n}")


def as_list(obj):
    if isinstance(obj, (tuple, list)) or obj is None:
        return obj
    return [obj]


def as_df(obj):
    if isinstance(obj, pd.Series):
        return pd.DataFrame(obj)
    return obj


def to_lower(data) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(data, pd.Series):
        try:
            return data.str.lower()
        except AttributeError:
            return data.copy()
    return data.apply(to_lower)


def drop_duplicates_case_insensitive(data, *, subset=None, keep="first"):
    if isinstance(data, pd.Series):
        return data[~data.str.lower().duplicated(keep)].copy(deep=True)
    dup_frame = to_lower(data)
    dups = dup_frame.duplicated(subset, keep)
    return data[~dups].copy(deep=True)


def clean_frame(frame, case_insensitive=True):
    if case_insensitive:
        return drop_duplicates_case_insensitive(frame).dropna(how="all")
    else:
        return frame.drop_duplicates().dropna(how="all")


@contextmanager
def disable_copy_warning():
    initial_setting, pd.options.mode.chained_assignment = pd.options.mode.chained_assignment, None
    try:
        yield
    finally:
        pd.options.mode.chained_assignment = initial_setting


def fillna_series(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(value=_NA_FILL_INTEGER)
    series, geo = fill_geoseries(series)
    if geo:
        return series
    return series.fillna(value=_NA_FILL_OBJECT)


def fillna(frame: Union[pd.DataFrame, pd.Series]):
    if isinstance(frame, pd.Series):
        return fillna_series(frame)
    for c, s in frame.items():
        frame.loc[:, c] = fillna_series(s)
    return frame


def unfillna_series(series: pd.Series):
    if pd.api.types.is_number(series):
        return series.replace(to_replace=_NA_FILL_INTEGER, value=pd.NA)
    return series.replace(to_replace=_NA_FILL_OBJECT, value=pd.NA)


def unfillna(frame: Union[pd.DataFrame, pd.Series]):
    if isinstance(frame, pd.Series):
        return unfillna_series(frame)
    for c, s in frame.items():
        frame.loc[:, c] = unfillna_series(s)
    return frame


def is_date_interval(interval: pd.Interval) -> bool:
    diff = interval.right - interval.left
    return (
        interval.left.strftime("%H%M%S%f") == "000000000000"
        and interval.right.strftime("%H%M%S%f") == "000000000000"
        and diff.delta % int(8.64e13) == 0
    )


def interval_to_range(interval: pd.Interval):
    if isinstance(interval.left, pd.Timestamp):
        if is_date_interval(interval):
            # date range
            return adapt(DateRange(interval.left.date(), interval.right.date()))
        try:
            if interval.left.tz is not None:
                return adapt(DateTimeTZRange(interval.left, interval.right))
        except AttributeError:
            pass
        return adapt(DateTimeRange(interval.left, interval.right))
    elif pd.api.types.is_number(interval.left):
        return adapt(NumericRange(interval.left, interval.right))
    else:
        raise TypeError("this adapter only works for dates/datetimes and numeric")


def period_to_interval(period: pd.Period) -> pd.Interval:
    return pd.IntervalIndex.from_arrays(
        period.start_time,
        period.end_time.values + 1,
        closed="left",
    )


def df_chunker(frame: Union[pd.DataFrame, pd.Series], size):
    for pos in range(0, len(frame), size):
        yield frame.iloc[pos : pos + size]


def to_date(ser: pd.Series):
    ser = ser.astype("Int64", copy=False).astype("str", copy=False)
    for n, fmt in DATE_FORMATS.items():
        msk = ser.str.len() == n
        ser[msk] = pd.to_datetime(ser[msk], format=fmt, cache=True)
    return ser.convert_dtypes()


def get_max_chars_in_common(s):
    lower_end = 1
    upper_end = s.str.len().min()

    def good(l):
        return len(s.str.slice(0, l).unique()) == 1

    if not good(lower_end):
        return None
    while True:
        midpoint = int(np.ceil((lower_end + upper_end) / 2))
        g = good(midpoint)
        if g:
            lower_end = midpoint
        else:
            upper_end = midpoint - 1
        if lower_end == upper_end:
            break
    return upper_end


def get_common_initial_str(s: pd.Series):
    if n := get_max_chars_in_common(s):
        return s.head(1).str.slice(0, n)[0]
    return None


def _infer_date(s: pd.Series):
    return all(isinstance(a, date) for a in s.dropna())


def convert_df_dates_to_timestamps(frame: pd.DataFrame):
    obj_cols = frame.select_dtypes("object").columns
    date_test = frame[obj_cols].apply(_infer_date)
    date_cols = date_test[date_test].index
    frame[date_cols] = frame[date_cols].apply(pd.to_datetime)
