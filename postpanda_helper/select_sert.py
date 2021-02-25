# coding=utf-8
import random
import string
from typing import Union

import numpy as np
import pandas as pd
from pandas import Series
from sqlalchemy import create_engine, MetaData, Table, types as satypes

from .psql_helpers import PandaCSVtoSQL


class SelectSert:
    _na_fill = (
        "na_value" + "".join(random.SystemRandom().choices(string.ascii_lowercase, k=6))
    ).lower()

    _na_fill_int = np.uint64(random.SystemRandom().getrandbits(64)).astype(np.int64)

    def __init__(self, connection, schema):
        if isinstance(connection, str):
            self._conn = create_engine(connection)
        else:
            self._conn = connection
        self._schema = schema
        self._tables = {}

    def _get_current_ids(self, table, force_update=False):
        if force_update or table not in self._tables:
            ret_ids = pd.read_sql_table(table, self._conn, schema=self._schema)
            ids = self._convert_dtypes_to_table(ret_ids, table)
            self._tables[table] = ids
        return self._tables[table]

    def _convert_dtypes_to_table(self, frame, table):
        meta = MetaData(self._conn)
        tbl = Table(table, meta, autoload=True, schema=self._schema)
        for c in frame.columns:
            tbl_col = tbl.columns[c]
            if isinstance(tbl_col.type, satypes.Integer):
                frame[c] = frame[c].astype("Int64")
            elif isinstance(tbl_col.type, satypes.Float):
                frame[c] = frame[c].astype("float")
            elif isinstance(tbl_col.type, satypes.Text):
                frame[c] = frame[c].astype("object")
            elif isinstance(tbl_col.type, satypes.Boolean):
                frame[c] = frame[c].astype("boolean")
        return frame

    def _insert(self, data, table):
        # print(data.shape)
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        sql_writer = PandaCSVtoSQL(
            data,
            table,
            self._conn,
            schema=self._schema,
            create_table=False,
            create_primary_key=False,
        )
        sql_writer.process_simple()

    def _fillmask(self, frame):
        fill_dict = frame.columns.to_series()
        num_cols = fill_dict.isin(frame.select_dtypes(include=np.number).columns)
        fill_dict[num_cols] = self._na_fill_int
        fill_dict[~num_cols] = self._na_fill
        return fill_dict, num_cols

    def fillna(self, frame: pd.DataFrame):
        fill_dict, num_cols = self._fillmask(frame)
        num_cols = frame.select_dtypes("number").columns
        frame[num_cols] = frame[num_cols].astype("Int64")
        return frame.fillna(value=fill_dict)

    def unfillna(self, frame: pd.DataFrame):
        fill_dict, _ = self._fillmask(frame)
        return frame.replace(to_replace=fill_dict, value=pd.NA)

    def get_ids(self, frame, table):
        sql_ids = self._get_current_ids(table)
        sql_tups = self.fillna(self.to_lower(sql_ids)[frame.columns]).apply(tuple, axis=1)
        df_tups = self.fillna(self.to_lower(frame).astype(sql_ids[frame.columns].dtypes)).apply(
            tuple, axis=1
        )
        to_remove = df_tups.isin(sql_tups)
        to_insert = frame[~to_remove]
        if to_insert.shape[0] == 0:
            return sql_ids
        self._insert(to_insert, table)
        return self._get_current_ids(table, force_update=True)

    def get_ids_simple(self, series, table, sql_col_name="code", case_insensitive=True):
        sql_ids = self._get_current_ids(table)
        sql_tups: Series = sql_ids[sql_col_name].fillna(self._na_fill)
        series_tups = series.astype(sql_tups.dtype).fillna(self._na_fill)
        if case_insensitive:
            to_remove = self.to_lower(series_tups).isin(self.to_lower(sql_tups))
        else:
            to_remove = series_tups.isin(sql_tups)

        to_insert = series[~to_remove]
        to_insert.name = sql_col_name
        if to_insert.shape[0] == 0:
            return sql_ids
        self._insert(to_insert, table)
        return self._get_current_ids(table, force_update=True)

    @classmethod
    def to_lower(cls, data) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(data, pd.Series):
            try:
                return data.str.lower()
            except AttributeError:
                return data.copy()
        return data.apply(cls.to_lower)

    @classmethod
    def drop_duplicates_case_insensitive(cls, data, *, subset=None, keep="first"):
        if isinstance(data, pd.Series):
            return data[~data.str.lower().duplicated(keep)].copy(deep=True)
        dup_frame = cls.to_lower(data)
        dups = dup_frame.duplicated(subset, keep)
        return data[~dups].copy(deep=True)

    @classmethod
    def clean_frame(cls, frame, case_insensitive=True):
        if case_insensitive:
            return cls.drop_duplicates_case_insensitive(frame).dropna(how="all")
        else:
            return frame.drop_duplicates().dropna(how="all")

    def replace_with_id(
        self,
        frame,
        columns,
        table_name,
        new_col_name=None,
        sql_col_name="code",
        sql_id_name="id",
        case_insensitive=True,
        delete_old=True,
    ):
        ids = self.get_ids_simple(
            self.clean_frame(frame[columns], case_insensitive),
            table_name,
            sql_col_name=sql_col_name,
            case_insensitive=case_insensitive,
        )
        idx_name = frame.index.name or "index"

        idx_name_fake = "fake_index_name"

        left = frame[[columns]].reset_index().astype("object")
        left.rename(columns={idx_name: idx_name_fake}, inplace=True)
        right = ids
        if case_insensitive:
            left = self.to_lower(left)
            right = self.to_lower(right)
        joined = pd.merge(
            left,
            right,
            how="left",
            left_on=columns,
            right_on=sql_col_name,
        )
        joined.set_index(idx_name_fake, inplace=True)
        joined.index.name = idx_name
        new_col_name = new_col_name or columns
        frame.loc[:, new_col_name] = self._downcast(joined.sort_index()[sql_id_name])
        if delete_old:
            frame.drop(columns=columns, inplace=True)

    @staticmethod
    def _downcast(data):
        if not pd.api.types.is_integer_dtype(data):
            return data
        dmax = data.fillna(0).max()
        for n in range(3, 7):
            nmax = np.iinfo(f"int{2 ** n}").max
            if dmax <= nmax:
                return data.astype(f"Int{2 ** n}")

    def replace_with_ids(
        self,
        frame: pd.DataFrame,
        columns,
        table_name,
        new_col_name,
        sql_column_names=None,
        delete_old=True,
        sql_id_name="id",
        case_insensitive=True,
    ):
        framecols = frame.reindex(columns=columns)
        if sql_column_names:
            framecols = framecols.rename(columns=dict(zip(columns, sql_column_names)))
            framecols[list(set(sql_column_names) - set(framecols.columns))] = None

        ids = self.get_ids(self.clean_frame(framecols, case_insensitive), table_name)
        sql_types = ids.loc[:, [c for c in ids.columns if c != sql_id_name]].dtypes
        idx_name = frame.index.name or "index"
        idx_name_fake = "fake_index_name"
        left = self.fillna(framecols.astype(sql_types).reset_index())

        left.rename(columns={idx_name: idx_name_fake}, inplace=True)
        # left.rename(columns={idx_name: idx_name_fake}, inplace=True)
        right = self.fillna(ids)
        if case_insensitive:
            left = self.to_lower(left)
            right = self.to_lower(right)
        joined = pd.merge(
            left,
            right,
            how="left",
            on=sql_column_names or columns,
        )
        joined = self.unfillna(joined)
        joined.set_index(idx_name_fake, inplace=True)
        joined.index.name = idx_name
        frame.loc[:, new_col_name] = self._downcast(joined.sort_index()[sql_id_name])
        if delete_old:
            frame.drop(columns=set(columns) & set(frame.columns), inplace=True)
