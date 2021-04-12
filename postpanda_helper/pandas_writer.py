# coding=utf-8
import asyncio
import gzip
import tempfile
from multiprocessing import Queue, Process
from pathlib import Path
from shutil import rmtree
from typing import Dict, Optional

import pandas as pd
from logger_mixin import LoggerMixin
from pathvalidate import sanitize_filepath
from sqlalchemy.engine import Engine

from .psql_helpers import PandaCSVtoSQL


class PandasWriter(LoggerMixin):
    """
    Multiprocessing writer that writes to CSV files before copying into DB
    """

    def __init__(self, engine):
        self._base_path = Path(tempfile.mkdtemp())
        self.engine = engine
        self._writers: Dict[(str, str), _PandasWriter] = {}

    def __getitem__(self, item):
        if not (
            isinstance(item, tuple) and len(item) == 2 and all(isinstance(i, str) for i in item)
        ):
            raise ValueError("Must specify as pw[schema,table]")
        return self.get_writer(item[0], item[1])

    def get_writer(self, schema, table):
        if (schema, table) in self._writers:
            return self._writers[(schema, table)]
        csv_file = self._base_path / f"{schema}.{table}.csv.bz2"
        csv_file = sanitize_filepath(csv_file, platform="auto")
        csv_file.touch()
        self._writers[(schema, table)] = _PandasWriter(csv_file, schema, table)
        return self._writers[(schema, table)]

    def join(self, upsert=True, cleanup=True):
        """Join and write SQL

        Args:
            upsert:  if we might need to overwrite things, insert everything
                into a Temp table, delete the existing rows, and then copy over
                to main table
            cleanup: should we delete the temporary files?

        Returns:

        """
        for n, w in self._writers.items():
            w.thread.join()
            w.to_sql(self.engine, upsert)
        if cleanup:
            rmtree(self._base_path, ignore_errors=True)

    def stop(self):
        """
        Done writing to queue, so close it out.
        Returns:

        """
        for n, w in self._writers.items():
            w.put_queue(None)


class _PandasWriter(LoggerMixin):
    def __init__(self, file_path: Path, schema: str, table: str) -> None:
        """

        Args:
            file_path: temporary directory to store the csv file in
            schema: schema in db
            table: table in db
        """
        super().__init__()

        self._path = file_path
        self._schema = schema
        self._table = table
        self._write_queue = Queue(maxsize=3)
        self._first_df = None

        self.thread = Process(target=self._writer)
        self.thread.start()

    def put_queue(self, df: Optional[pd.DataFrame]) -> None:
        """Add dataframe to queue

        Args:
            df:

        Returns:

        """
        if self._first_df is None:
            self._first_df = df.head(100)
        self._write_queue.put(df)

    def _get_file(self, writeable=True):
        """
        Open up a file
        Args:
            writeable:

        Returns:

        """
        if writeable:
            mode = "at"
        else:
            mode = "rt"
        return gzip.open(self._path, mode, compresslevel=1, encoding="utf8", newline="\n")

    def _writer(self):
        with self._get_file() as fh:
            # Call to_write.get() until it returns None
            while (df := self._write_queue.get()) is not None:
                self.logger.debug(
                    f"Writing {self._schema}.{self._table} of size {df.shape} "
                    f"Queue:{self._write_queue.qsize()}"
                )
                df.to_csv(fh, header=False, index=False, line_terminator="\n")

    def to_sql(self, engine: Engine, upsert: bool = True) -> None:
        """

        Args:
            engine: Engine to connect to
            upsert: if we might need to overwrite things, insert everything
             into a Temp table, delete the existing rows, and then copy over
             to main table

        Returns:

        """
        sql_writer = PandaCSVtoSQL(
            self._first_df,
            self._table,
            engine,
            # primary_key=self.primary_key,
            schema=self._schema,
            create_table=False,
            create_primary_key=False,
        )
        self.logger.debug(f"writing {self._path.name} to db")
        with self._get_file(False) as fh:
            if upsert:
                asyncio.run(sql_writer.import_csv(fh))
            else:
                sql_writer.import_csv_no_temp(fh)
