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

#  dictionary of frequency,  mask function, kwargs
DATE_FREQ_MAP = {"A": "Y", "Q": "Q", "M": "M", "4": "W", "W": "W", "D": "D"}


def to_string_tuples(frame):
    """Converts frame columns to combined tuple string"""
    rows = []
    for r in frame.itertuples(index=False):
        row = [str(a) if pd.notna(a) else "" for a in r]
        if all([a == "" for a in row]):
            rows.append(pd.NA)
        else:
            rows.append(f'({",".join(row)})')

    return pd.Series(
        rows,
        index=frame.index,
    )


def value_to_nan(frame, value=0.0):
    return frame.replace(value, pd.NA)


def to_string_tuples_drop_val(frame, value=0.0):
    return to_string_tuples(value_to_nan(frame, value))


def strip_all_spaces(frame: Union[pd.Series, pd.DataFrame]):
    """strips all extra spaces (equivalent to a trim and remove doubled spaces)

    Args:
        frame:

    Returns:

    """

    if isinstance(frame, pd.Series):
        if pd.api.types.is_object_dtype(frame):
            return frame.str.replace(" +", " ", regex=True).str.strip()
        else:
            return frame
    return frame.apply(strip_all_spaces)


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


def map_to_bool(series: pd.Series, true_val=None, false_val=None, fillna=False):
    rep_map = {}
    if true_val is not None:
        rep_map[true_val] = True
    if false_val is not None:
        rep_map[false_val] = False
    series = series.replace(rep_map)
    if fillna is not None:
        series = series.fillna(fillna)
    return series


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


def ser_to_date(ser: pd.Series, freq, start_or_end=None):
    if start_or_end is not None:
        if start_or_end.lower().startswith("start"):
            start_or_end = "start_time"
        elif start_or_end.lower().startswith("end"):
            start_or_end = "end_time"
        else:
            raise ValueError(f"Unknown start_or_end {start_or_end}")
    else:
        start_or_end = "end_time"
    if pd.api.types.is_float_dtype(ser):
        ser = ser.astype("Int64")
    pdindx = getattr(pd.PeriodIndex(ser, freq=freq), start_or_end)

    return pd.to_datetime(pdindx.date).to_series(index=ser.index)


def to_date(frame: pd.DataFrame, freq_col, date_col, start_or_end=None, date_freq_map=None):
    if date_freq_map is None:
        date_freq_map = DATE_FREQ_MAP
    for f, ser in frame.groupby(freq_col)[date_col]:
        frame.loc[ser.index, date_col] = ser_to_date(
            ser, freq=date_freq_map[f], start_or_end=start_or_end
        )
    frame.loc[:, date_col] = frame.loc[:, date_col].convert_dtypes()


def get_max_chars_in_common(s):
    lower_end = 1
    upper_end = s.str.len().min()

    def is_good(num_chars):
        return len(s.str.slice(0, num_chars).unique()) == 1

    if not is_good(lower_end):
        return None
    while True:
        midpoint = int(np.ceil((lower_end + upper_end) / 2))
        g = is_good(midpoint)
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
    if not len(s.dropna()):
        return False
    return all(isinstance(a, date) for a in s.dropna())


def _to_datetime(s: pd.Series):
    has_tzinfo = s.apply(lambda x: getattr(x, "tzinfo", None) is not None).any()
    if has_tzinfo:
        return pd.to_datetime(s, utc=True)
    return pd.to_datetime(s)


def convert_df_dates_to_timestamps(frame: pd.DataFrame, inplace=False):
    obj_cols = frame.select_dtypes("object").columns
    date_test = frame[obj_cols].apply(_infer_date)
    date_cols = date_test[date_test].index

    if inplace:
        frame[date_cols] = frame[date_cols].apply(_to_datetime)
    else:
        ff = frame.copy()
        ff[date_cols] = ff[date_cols].apply(_to_datetime)
        return ff
