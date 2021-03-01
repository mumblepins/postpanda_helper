# coding=utf-8
import random
import string
from contextlib import contextmanager
from typing import Union

import numpy as np
import pandas as pd
from psycopg2._psycopg import adapt
from psycopg2._range import DateRange, DateTimeTZRange, DateTimeRange, NumericRange

_NA_FILL_INTEGER = np.uint64(random.SystemRandom().getrandbits(64)).astype(np.int64)
_NA_FILL_OBJECT = "na_val_" + "".join(random.SystemRandom().choices(string.ascii_lowercase, k=8))


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
    if pd.api.types.is_number(series):
        return series.fillna(value=_NA_FILL_INTEGER)
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
