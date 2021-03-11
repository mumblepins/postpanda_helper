# coding=utf-8
import io
import warnings
from contextlib import contextmanager
from typing import Optional, MutableMapping, Any

import pandas as pd
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.sql import SQLTable, pandasSQL_builder
from psycopg2 import errorcodes, sql
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert, DATERANGE, TSTZRANGE, TSRANGE, NUMRANGE
from sqlalchemy.engine.base import Connection
from sqlalchemy.exc import ProgrammingError, SAWarning
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.ddl import CreateColumn

from .geo_helpers import convert_geometry_for_postgis
from .pd_helpers import is_date_interval


@compiles(CreateColumn, "postgresql")
def use_identity(element, compiler, **kw):
    text = compiler.visit_create_column(element, **kw)
    text = text.replace("SERIAL", "INT")
    return text


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def create_df_table_altered(
    frame,
    name,
    con,
    primary_keys,
    schema=None,
    index=True,
    index_label=None,
    dtype=None,
):
    pandas_sql = pandasSQL_builder(con, schema=schema)
    if dtype and not is_dict_like(dtype):
        dtype = {col_name: dtype for col_name in frame}

    if dtype is not None:
        from sqlalchemy.types import to_instance, TypeEngine

        for col, my_type in dtype.items():
            if not isinstance(to_instance(my_type), TypeEngine):
                raise ValueError(f"The type of {col} is not a SQLAlchemy type")

    table = SQLTable(
        frame=frame,
        name=name,
        pandas_sql_engine=pandas_sql,
        schema=schema,
        index=index,
        index_label=index_label,
        dtype=dtype,
        keys=conform_to_list(primary_keys),
        if_exists="append",
    )
    try:
        table.create()
        table.insert(method=psql_upsert(primary_keys))
    except ProgrammingError as e:
        print("Warning, recreating table due to column mismatch")
        if errorcodes.lookup(e.orig.pgcode) == "UNDEFINED_COLUMN":
            table.if_exists = "replace"
            table.create()
            table.insert(method=psql_upsert(primary_keys))
        else:
            raise


class PandaCSVtoSQL:
    def get_full_tablename(self, quotes=True):
        if quotes:
            return (
                f'"{self.schema}"."{self.table}"' if self.schema is not None else f'"{self.table}"'
            )
        else:
            return f"{self.schema}.{self.table}" if self.schema is not None else f"{self.table}"

    def get_col_names(self):
        cols = list(self.dframe.columns)
        if self.index:
            cols = [self.dframe.index.name] + cols
        return cols

    @property
    def sa_table(self):
        if self._sa_table is None:
            with disable_reflection_warning():
                self._sa_table = Table(
                    self.table, MetaData(self.engine), autoload=True, schema=self.schema
                )
        return self._sa_table

    @property
    def primary_key(self):
        return [c.name for c in self.sa_table.primary_key.columns]

    def __init__(
        self,
        dframe: pd.DataFrame,
        table,
        engine=None,
        primary_key=None,
        schema=None,
        chunksize=1000000,
        index=False,
        create_table=True,
        create_primary_key=True,
        **kwargs,
    ):
        self.engine = engine
        self.dframe = dframe
        self.chunksize = chunksize
        self.index = index
        self.table = table
        self._sa_table = None
        self.schema = schema
        # if primary_key:
        #     self.primary_key = conform_to_list(primary_key)
        self.full_tablename = self.get_full_tablename()
        self.temporary_tablename = "temp_table"
        self.cols = self.get_col_names()
        self.sql_cols = sql.Composed([sql.Identifier(c) for c in self.cols]).join(", ")
        self.csv_import_sql = sql.SQL(
            "COPY {table} ({columns}) FROM STDIN WITH (FORMAT CSV, FORCE_NULL ({columns}))"
        )
        if create_table and engine and primary_key:
            create_df_table_altered(
                dframe.head(1),
                table,
                engine,
                primary_keys=primary_key,
                schema=schema,
                index=index,
                **kwargs,
            )

        # Check if primary key exists; if not, create it
        if engine:
            meta = MetaData()
            with disable_reflection_warning():
                tt = Table(table, meta, schema=schema, autoload=True, autoload_with=engine)
            if create_primary_key:
                if len(tt.primary_key) == 0:
                    with engine.connect() as conn:
                        # @formatter:off
                        conn.execution_options(autocommit=True).execute(
                            sql.SQL("ALTER TABLE {TABLE} ADD PRIMARY KEY ({COLUMNS})").format(
                                table=sql.SQL(self.full_tablename),
                                columns=sql.Composed([sql.Identifier(c) for c in primary_key]).join(
                                    ", "
                                ),
                            )
                        )
                        # @formatter:on
            self.temp_table = Table(self.temporary_tablename, meta, prefixes=["TEMPORARY"])
            for column in tt.columns:
                self.temp_table.append_column(column.copy())

    async def to_csv(self, frame=None, file_handle=None, seek=True):
        if frame is None:
            frame = self.dframe
        if file_handle is None:
            file_handle = io.StringIO(newline="\n")
        if seek:
            start_pos = file_handle.seek(0, io.SEEK_END)
        frame.to_csv(file_handle, header=False, index=self.index, line_terminator="\n")
        if seek:
            file_handle.seek(start_pos)
        return file_handle

    def _to_csv_simple(self):
        file_handle = io.StringIO(newline="\n")
        self.dframe.to_csv(file_handle, header=False, index=self.index, line_terminator="\n")
        file_handle.seek(0)
        return file_handle

    def import_csv_no_temp(self, csv_fh):
        with self.engine.connect() as conn:
            csv_table = sql.SQL(self.full_tablename)
            import_sql_command = self.csv_import_sql.format(
                table=csv_table,
                columns=self.sql_cols,
            )
            with conn.connection.cursor() as cur:
                cur.copy_expert(import_sql_command, csv_fh)
                conn.connection.commit()

    async def import_csv(self, csv_fh, use_temp_table=True, wait_on=None):
        with self.engine.connect() as conn:
            if use_temp_table:
                self.temp_table.create(conn)
                csv_table = sql.SQL(self.temporary_tablename)
            else:
                csv_table = sql.SQL(self.full_tablename)
            import_sql_command = self.csv_import_sql.format(
                table=csv_table,
                columns=self.sql_cols,
            )
            with conn.connection.cursor() as cur:
                cur.copy_expert(import_sql_command, csv_fh)
                if use_temp_table:
                    matchers = [
                        f'{self.full_tablename}."{k}" = {self.temporary_tablename}."{k}"'
                        for k in self.primary_key
                    ]
                    cur.execute(
                        f"DELETE FROM {self.full_tablename} "
                        f"USING {self.temporary_tablename} "
                        f'WHERE {" and ".join(matchers)}'
                    )
                    if wait_on is not None:
                        await wait_on
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO " + "{table_name} ({cols}) SELECT {cols} FROM {t_table}"
                        ).format(
                            table_name=sql.SQL(self.full_tablename),
                            cols=sql.Composed([sql.Identifier(c) for c in self.cols]).join(", "),
                            t_table=sql.SQL(self.temporary_tablename),
                        )
                    )
                    self.temp_table.drop(conn)
                conn.connection.commit()

    async def process(self, use_temp_table=True, wait_on=None):
        for cdf in chunker(self.dframe, self.chunksize):
            tfile = await self.to_csv(cdf)
            await self.import_csv(tfile, use_temp_table=use_temp_table, wait_on=wait_on)

    def process_simple(self):
        fh = self._to_csv_simple()
        self.import_csv_no_temp(fh)


def conform_to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj


def psql_upsert(index_elements):
    index_elements = conform_to_list(index_elements)

    def ret_func(pdtable: SQLTable, conn: Connection, keys, data_iter):
        data = [dict(zip(keys, row)) for row in data_iter]

        insert_stmt = insert(pdtable.table, data)
        upsert_set = {k: insert_stmt.excluded[k] for k in keys if k not in index_elements}
        if len(upsert_set) == 0:
            do_update = insert_stmt.on_conflict_do_nothing(index_elements=index_elements)
        else:
            do_update = insert_stmt.on_conflict_do_update(
                index_elements=index_elements, set_=upsert_set
            )
        conn.execute(do_update)

    return ret_func


def possible_upsert(pdtable: SQLTable, conn: Connection, keys, data_iter):
    tbl = pdtable.table
    with disable_reflection_warning():
        tbl._autoload(tbl.metadata, conn.engine, None)
    data = [dict(zip(keys, row)) for row in data_iter]
    unique_cols = [c for i in tbl.indexes if i.unique for c in i.columns]
    index_elements = [c.name for c in unique_cols]
    if len(index_elements) == 0:
        index_elements = [c.name for c in tbl.primary_key.columns]
    insert_stmt = insert(pdtable.table, data)
    upsert_set = {k: insert_stmt.excluded[k] for k in keys if k not in index_elements}
    if len(upsert_set) == 0:
        do_update = insert_stmt.on_conflict_do_nothing(index_elements=index_elements)
    else:
        do_update = insert_stmt.on_conflict_do_update(
            index_elements=index_elements, set_=upsert_set
        )
    conn.execute(do_update)


@contextmanager
def disable_reflection_warning():
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=SAWarning, message="Skipped unsupported reflection.*"
            )
            yield
    finally:
        pass


def pd_to_sql(
    frame: pd.DataFrame,
    name: str,
    con,
    schema: Optional[str] = None,
    if_exists="fail",
    index=True,
    index_label=None,
    chunksize=None,
    dtype: Optional[MutableMapping[str, Any]] = None,
    method=None,
):
    """
    With support for daterange and tsrange

    """
    dtm = dtype or {}
    copied = False
    columns = frame.columns
    for c in columns:
        if c in dtm:
            continue
        if pd.api.types.is_interval_dtype(frame[c]):
            ex = frame[c][frame[c].first_valid_index()]
            if isinstance(ex.left, pd.Timestamp):
                if is_date_interval(ex):
                    dtm[c] = DATERANGE
                elif hasattr(ex, "tz") and ex.tz is not None:
                    dtm[c] = TSTZRANGE
                else:
                    dtm[c] = TSRANGE
            elif pd.api.types.is_number(ex.left):
                dtm[c] = NUMRANGE
        _frame, _dmap = convert_geometry_for_postgis(frame, c, not copied)
        if _frame is not None:
            frame = _frame
        dtm.update(_dmap)
        if not copied:
            copied = not copied
        # elif HAS_GEO_EXTENSIONS and isinstance(frame[c].dtype, GeometryDtype):
        #     # copy because we're going to mess with the column
        #     if not copied:
        #         frame = frame.copy()
        #     s = GeoSeries(frame[c])
        #     try:
        #         geometry_type, has_curve = get_geometry_type(s)
        #     except ValueError:
        #         frame[c] = frame[c].astype("object")
        #         continue
        #     srid = s.crs.to_epsg(min_confidence=25) or -1
        #     dtm[c] = Geometry(geometry_type=geometry_type, srid=srid)
        #     frame[c] = geometry_to_ewkb(frame[c], srid)

    dtm = dtm or None
    return frame.to_sql(
        name=name,
        con=con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtm,
        method=method,
    )
