# coding=utf-8
import random
import string
from typing import Union, Sequence, Mapping, Any, Optional

import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, types as satypes
from sqlalchemy.engine import Engine

from . import pd_helpers as pdh
from .psql_helpers import PandaCSVtoSQL


class SelectSert:
    _IDX_NAME_FAKE = "fake_index_name_" + "".join(
        random.SystemRandom().choices(string.ascii_lowercase, k=6)
    )

    def __init__(self, connection: Union[str, Engine], schema: str):
        """

        Args:
            connection: Connection string or sqlalchemy Engine
            schema: schema where the lookup tables are located
        """
        if isinstance(connection, str):
            self._conn = create_engine(connection)
        else:
            self._conn = connection
        self._schema = schema
        self._tables = {}

    def _get_current_ids(self, table: str, force_update=False) -> pd.DataFrame:
        if force_update or table not in self._tables:
            ret_ids = pd.read_sql_table(table, self._conn, schema=self._schema)
            ids = self._convert_dtypes_to_table(ret_ids, table)
            self._tables[table] = ids
        return self._tables[table]

    def _convert_dtypes_to_table(self, frame: pd.DataFrame, table: str) -> pd.DataFrame:
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

    def _insert(self, data: Union[pd.Series, pd.DataFrame], table: str) -> None:
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

    def _get_ids_frame(
        self, frame: pd.DataFrame, table: str, case_insensitive: bool = True
    ) -> pd.DataFrame:
        sql_ids = self._get_current_ids(table)
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
        to_remove = df_tups.isin(sql_tups)
        to_insert = frame[~to_remove]
        if to_insert.shape[0] == 0:
            return sql_ids
        self._insert(to_insert, table)
        return self._get_current_ids(table, force_update=True)

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
            with pdh.disable_copy_warning():
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
        ids = self._get_ids_frame(pdh.clean_frame(frame_cols, case_insensitive), table_name)
        sql_types = ids.loc[:, [c for c in ids.columns if c != sql_id_name]].dtypes

        left = pdh.fillna(
            frame_cols.astype(sql_types[sql_types.keys() & frame_cols.columns]).reset_index()
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
