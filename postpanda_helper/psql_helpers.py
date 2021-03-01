# coding=utf-8
import io

import pandas as pd
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.sql import SQLTable, pandasSQL_builder
from psycopg2 import errorcodes, sql
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine.base import Connection
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.ddl import CreateColumn


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
        self.schema = schema
        if primary_key:
            self.primary_key = conform_to_list(primary_key)
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


#
# async def panda_to_sql_via_csv(dframe: pd.DataFrame,
#                                table,
#                                engine: Engine,
#                                primary_key,
#                                schema=None,
#                                chunksize=1000000,
#                                index=False,
#                                create_table=True,
#                                create_primary_key=True,
#                                **kwargs):
#     conn: Connection
#     cur: cursor
#     primary_key = conform_to_list(primary_key)
#     full_tablename = f'"{schema}"."{table}"' if schema is not None else f'"{table}"'
#     temporary_tablename = 'temp_table'
#     # trunc_sql = f'TRUNCATE TABLE {full_tablename}'
#     cols = list(dframe.columns)
#     if index:
#         cols = [dframe.index.name] + cols
#     csv_import_sql = sql.SQL('COPY {table} ({columns}) FROM STDIN WITH (FORMAT CSV, FORCE_NULL ({columns}))').format(
#         table=sql.SQL(temporary_tablename),
#         columns=sql.Composed([sql.Identifier(c) for c in cols]).join(', '))
#     # method = psql_upsert(primary_key)
#     # lazily check and make sure the table conforms to our dataframe (all columns that we want to insert exist)
#     # WARNING: deletes and recreates if new column, doesn't ALTER table to create column
#     # TODO: Use ALTER statement to add column?
#     if create_table:
#         create_df_table_altered(dframe.head(1), table, engine, primary_keys=primary_key, schema=schema, index=index,
#                                 **kwargs)
#
#     # Check if primary key exists; if not, create it
#     meta = MetaData()
#     tt = Table(table, meta, schema=schema, autoload=True, autoload_with=engine)
#     if create_primary_key and len(tt.primary_key) == 0:
#         with engine.connect() as conn:
#             # @formatter:off
#             conn.execution_options(autocommit=True).execute(
#                 sql.SQL('ALTER TABLE {TABLE} ADD PRIMARY KEY ({COLUMNS})').format(
#                     table=sql.SQL(full_tablename),
#                     columns=sql.Composed([sql.Identifier(c) for c in primary_key]).join(', '))
#             )
#             # @formatter:on
#     temp_table = Table(temporary_tablename, meta, prefixes=['TEMPORARY'])
#     for column in tt.columns:
#         temp_table.append_column(column.copy())
#
#     # with tqdm(total=len(dframe)) as pbar:
#     for cdf in chunker(dframe, chunksize):
#         # fh, tfile = tempfile.mkstemp(suffix='.csv')
#         tfile = io.StringIO(newline='\n')
#         cdf.to_csv(tfile, header=False, index=index, line_terminator='\n')
#         # os.chmod(tfile, 0o0644)
#
#         # os.close(fh)
#         tfile.seek(0)
#         with engine.connect() as conn:
#             temp_table.create(conn)
#
#             with conn.connection.cursor() as cur:
#                 cur.copy_expert(csv_import_sql, tfile)
#                 matchers = [f'{full_tablename}."{k}" = {temporary_tablename}."{k}"' for k in primary_key]
#                 cur.execute(
#                     f'DELETE FROM {full_tablename} '
#                     f'USING {temporary_tablename} '
#                     f'WHERE {" and ".join(matchers)}'
#                 )
#
#                 cur.execute(sql.SQL('INSERT INTO ' +
#                                     '{table_name} ({cols}) SELECT {cols} FROM {t_table}').format(
#                     table_name=sql.SQL(full_tablename),
#                     cols=sql.Composed([sql.Identifier(c) for c in cols]).join(', '),
#                     t_table=sql.SQL(temporary_tablename)
#                 ))
#                 # cur.execute(f'DROP TABLE {temporary_tablename}')
#             temp_table.drop(conn)
#             conn.connection.commit()
#             # con.execution_options(autocommit=True).execute(copy_sql.format(tfile))
#         # os.remove(tfile)
#         # pbar.update(len(cdf))
#
#
# def _df_to_sql_chunk_writer(cdf, index, engine):
#     tfile = io.StringIO(newline='\n')
#     cdf.to_csv(tfile, header=False, index=index, line_terminator='\n')
#     # os.chmod(tfile, 0o0644)
#
#     # os.close(fh)
#     tfile.seek(0)
#     with engine.connect() as conn:
#         temp_table.create(conn)
#
#         with conn.connection.cursor() as cur:
#             cur.copy_expert(csv_import_sql, tfile)
#             matchers = [f'{full_tablename}."{k}" = {temporary_tablename}."{k}"' for k in primary_key]
#             cur.execute(
#                 f'DELETE FROM {full_tablename} '
#                 f'USING {temporary_tablename} '
#                 f'WHERE {" and ".join(matchers)}'
#             )
#
#             cur.execute(sql.SQL('INSERT INTO ' +
#                                 '{table_name} ({cols}) SELECT {cols} FROM {t_table}').format(
#                 table_name=sql.SQL(full_tablename),
#                 cols=sql.Composed([sql.Identifier(c) for c in cols]).join(', '),
#                 t_table=sql.SQL(temporary_tablename)
#             ))
#             # cur.execute(f'DROP TABLE {temporary_tablename}')
#         temp_table.drop(conn)
#         conn.connection.commit()
#


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
