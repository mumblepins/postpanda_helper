# coding=utf-8
import random
import string
from collections import defaultdict
from typing import Union, Sequence, Mapping, Any, Optional

import pandas as pd
import sqlalchemy.sql as sasql
from sqlalchemy import create_engine, MetaData, Table, types as satypes, and_
from sqlalchemy.engine import Engine

from . import pd_helpers as pdh
from .geo_helpers import df_to_shape
from .pd_helpers import get_common_initial_str, convert_df_dates_to_timestamps
from .psql_helpers import possible_upsert, disable_reflection_warning, pd_to_sql


class SelectSert:
    _IDX_NAME_FAKE = "fake_index_name_" + "".join(
        random.SystemRandom().choices(string.ascii_lowercase, k=6)
    )

    def __init__(self, connection: Union[str, Engine], default_schema: Optional[str] = None):
        """

        Args:
            connection: Connection string or sqlalchemy Engine
            default_schema: schema where the lookup tables are located
        """
        if isinstance(connection, str):
            self._conn = create_engine(connection)
        else:
            self._conn = connection
        self._schema = default_schema
        self._tables = defaultdict(dict)
        self._sa_tables = defaultdict(dict)
        self._meta = MetaData(self._conn)

    def _reflect_table(self, schema, table):
        if table not in self._sa_tables:
            with disable_reflection_warning():
                self._sa_tables[schema][table] = Table(
                    table, self._meta, schema=schema, autoload=True
                )
        return self._sa_tables[schema][table]

    def _get_current_ids(
        self,
        table: str,
        force_update=False,
        schema=None,
        filter_map: Optional[Mapping[filter, Any]] = None,
    ) -> pd.DataFrame:
        schema = schema or self._schema
        table_lkp = table
        if filter_map:
            table_lkp = (table, tuple(filter_map.items()))
        if force_update or table_lkp not in self._tables[schema]:
            tt = self._reflect_table(schema, table)
            with disable_reflection_warning():
                if filter_map:
                    query = sasql.select(tt.columns).where(
                        and_(*[sasql.column(k).like(f"{v}%") for k, v in filter_map.items()])
                    )

                    ret_ids = pd.read_sql_query(query, self._conn)
                else:
                    ret_ids = pd.read_sql_table(tt.name, self._conn, schema=tt.schema)
                df_to_shape(tt, ret_ids)
            ids = self._convert_dtypes_to_table(ret_ids, table, schema=schema)
            self._tables[schema][table_lkp] = ids
        return self._tables[schema][table_lkp]

    def _convert_dtypes_to_table(
        self, frame: pd.DataFrame, table: str, schema=None
    ) -> pd.DataFrame:
        meta = MetaData(self._conn)
        schema = schema or self._schema
        with disable_reflection_warning():
            tbl = Table(table, meta, autoload=True, schema=schema)
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

    def _insert(self, data: Union[pd.Series, pd.DataFrame], table: str, schema=None) -> None:

        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        schema = schema or self._schema

        pd_to_sql(
            data,
            table,
            con=self._conn,
            schema=schema,
            if_exists="append",
            index=False,
            method=possible_upsert,
        )
        # sql_writer = PandaCSVtoSQL(
        #     data,
        #     table,
        #     self._conn,
        #     schema=schema,
        #     create_table=False,
        #     create_primary_key=False,
        # )
        # sql_writer.process_simple()

    def many_many_link(self, frame, columns, table, schema=None):
        schema = schema or self._schema
        data = frame[columns]
        list_cols = [k for k, c in data.items() if isinstance(c.dropna().iloc[0], (tuple, list))]
        for c in list_cols:
            data = data.explode(c)
        data.dropna(inplace=True)
        data.to_sql(
            table,
            con=self._conn,
            schema=schema,
            if_exists="append",
            index=False,
            method=possible_upsert,
        )

    def _get_ids_frame(
        self,
        frame: pd.DataFrame,
        table: str,
        case_insensitive: bool = True,
        schema: Optional[str] = None,
        filter_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:

        if filter_columns:
            filter_columns = {c: get_common_initial_str(frame[c]) for c in filter_columns}
        sql_ids = self._get_current_ids(table, schema=schema, filter_map=filter_columns)
        schema = schema or self._schema
        if case_insensitive:
            test_sql = pdh.to_lower(sql_ids)
            test_frame = pdh.to_lower(frame)
        else:
            test_sql = sql_ids
            test_frame = frame
        # noinspection PyTypeChecker
        sql_tups = pdh.fillna(test_sql[frame.columns]).apply(tuple, axis=1)
        # noinspection PyTypeChecker
        df_tups = pdh.fillna(test_frame.astype(sql_ids[frame.columns].dtypes)).apply(tuple, axis=1)
        # to_remove = pd.Series([v in sql_tups.to_list() for v in df_tups], index=df_tups.index)
        to_remove = df_tups.isin(sql_tups)
        to_insert = frame[~to_remove]
        if to_insert.shape[0] == 0:
            return sql_ids
        self._insert(to_insert, table, schema=schema)
        return self._get_current_ids(
            table, force_update=True, schema=schema, filter_map=filter_columns
        )

    def _unset_index(self, frame: pd.DataFrame, left: pd.DataFrame) -> str:
        idx_name = frame.index.name or "index"
        left.rename(columns={idx_name: self._IDX_NAME_FAKE}, inplace=True)
        return idx_name

    def _set_index_and_join(
        self,
        frame: pd.DataFrame,
        joined: pd.DataFrame,
        idx_name: str,
        new_col_name: str,
        sql_id_name: str,
    ) -> None:
        try:
            joined.set_index(self._IDX_NAME_FAKE, inplace=True)
        except AttributeError:
            pass
        joined.index.name = idx_name
        frame.loc[:, new_col_name] = pdh.downcast(pdh.as_df(joined)[sql_id_name])

    def replace_multiple(
        self,
        frame: pd.DataFrame,
        replace_spec: Sequence[Mapping[str, Any]],
    ) -> None:
        for rs in replace_spec:
            self.replace_with_ids(frame, **rs)

    def replace_with_ids(
        self,
        frame: pd.DataFrame,
        columns: Union[Sequence[str], str],
        table_name: str,
        new_col_name: str,
        *,
        sql_column_names: Optional[Union[Sequence[str], str]] = None,
        sql_id_name: str = "id",
        case_insensitive: bool = True,
        delete_old: bool = True,
        sql_columns_notnull: Optional[Sequence[str]] = None,
        column_split: Optional[Mapping[str, str]] = None,
        schema: Optional[str] = None,
        filter_columns: Optional[Sequence[str]] = None,
    ) -> None:
        with pdh.disable_copy_warning():
            self._replace_with_ids(
                frame=frame,
                columns=columns,
                table_name=table_name,
                new_col_name=new_col_name,
                sql_column_names=sql_column_names,
                sql_id_name=sql_id_name,
                case_insensitive=case_insensitive,
                delete_old=delete_old,
                sql_columns_notnull=sql_columns_notnull,
                column_split=column_split,
                schema=schema,
                filter_columns=filter_columns,
            )

    def _replace_with_ids(
        self,
        frame: pd.DataFrame,
        columns: Union[Sequence[str], str],
        table_name: str,
        new_col_name: str,
        *,
        sql_column_names: Optional[Union[Sequence[str], str]] = None,
        sql_id_name: str = "id",
        case_insensitive: bool = True,
        delete_old: bool = True,
        sql_columns_notnull: Optional[Sequence[str]] = None,
        column_split: Optional[Mapping[str, str]] = None,
        schema: Optional[str] = None,
        filter_columns: Optional[Sequence[str]] = None,
    ) -> None:
        columns = pdh.as_list(columns)
        sql_column_names = pdh.as_list(sql_column_names)
        frame_cols = frame.reindex(columns=columns)
        if column_split:
            for k, v in column_split.items():
                if frame_cols[k].dropna().any():
                    frame_cols[k] = frame_cols[k].str.split(v)
        if sql_column_names:
            frame_cols = frame_cols.rename(columns=dict(zip(columns, sql_column_names)))
            frame_cols.loc[:, list(set(sql_column_names) - set(frame_cols.columns))] = None

        list_cols = frame_cols.applymap(lambda x: isinstance(x, list)).any()
        have_lists = None
        if list_cols.any():
            have_lists = list_cols[list_cols].index
            for c in have_lists:
                frame_cols = frame_cols.explode(c)

        if sql_columns_notnull:
            frame_cols.dropna(subset=sql_columns_notnull, how="any", inplace=True)
        if len(frame_cols) == 0:
            return
        ids = self._get_ids_frame(
            pdh.clean_frame(frame_cols, case_insensitive),
            table_name,
            schema=schema,
            filter_columns=filter_columns,
        )
        sql_types = ids.loc[:, [c for c in ids.columns if c != sql_id_name]].dtypes

        frame_cols = convert_df_dates_to_timestamps(frame_cols)
        ids = convert_df_dates_to_timestamps(ids)
        left = pdh.fillna(
            frame_cols.astype(
                sql_types[sql_types.index.intersection(frame_cols.columns)]
            ).reset_index()
        )
        idx_name = self._unset_index(frame, left)
        right = pdh.fillna(ids)
        if case_insensitive:
            left[sql_column_names or columns] = pdh.to_lower(left[sql_column_names or columns])
            right[sql_column_names or columns] = pdh.to_lower(right[sql_column_names or columns])
        joined = pd.merge(
            left,
            right,
            how="left",
            on=sql_column_names or columns,
        )
        if have_lists is not None:
            joined = joined.groupby(self._IDX_NAME_FAKE)[sql_id_name].apply(list)
        self._set_index_and_join(frame, joined, idx_name, new_col_name, sql_id_name)
        if delete_old:
            frame.drop(columns=set(columns) & set(frame.columns), inplace=True)
