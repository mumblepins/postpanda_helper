# coding=utf-8
# coding=utf-8
from hashlib import blake2b
from multiprocessing import Process
from multiprocessing import Queue
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
from logger_mixin import LoggerMixin


class PandasReader(LoggerMixin, Process):
    """Multiprocess based CSV reader

    Can also either convert a column to a UUID (128 bit, hex encoded)
    by simply writing it as a zero padded hex value, or can calculate a
    checksum (As a UUID) for the whole line to use as a unique primary key.

    The checksum is started with the input filename and uses the :func:`~hashlib.blake2b`
    hash algo.

    Args:
        *args: positional arguments to pass to :func:`pandas.read_csv`
        queue_max_size: max queue size
        get_checksum: should we find the checksum of the line?
        column_to_uuid: which column should we convert to an UUID?

            ``{'original_column': 'original_col_name', 'new_column': 'new_col_name'}``
        **kwargs: keyword arguments to pass to :func:`pandas.read_csv`
    """

    def __init__(
        self,
        *args,
        queue_max_size: int = 3,
        get_checksum: bool = False,
        column_to_uuid: Optional[Mapping[str, str]] = None,
        **kwargs,
    ):
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        self._q = Queue(maxsize=queue_max_size)
        self._get_checksum = get_checksum
        try:
            name = Path(args[0]).name
        except:
            name = Path(kwargs.get("filepath_or_buffer", "")).name
        self.fname = name.encode("utf8")
        self._column_to_uuid = column_to_uuid

    @property
    def queue(self) -> Queue:
        """

        Returns: A queue containing the chunks that have been read

        """
        return self._q

    def run(self):
        if self._get_checksum:
            return self._run_w_cs()
        else:
            return self._run_no_cs()

    def _run_no_cs(self):
        pd_reader = pd.read_csv(*self._args, **self._kwargs)
        for df_chunk in pd_reader:
            self.logger.debug(f"Read into queue: {df_chunk.index.max() + 1}")
            self._finish_chunk(df_chunk)
        self._q.put(None)

    def _checksum_line(self, line):
        hasher = blake2b(self.fname, digest_size=16)
        hasher.update(f"{line.name:012d}".encode("utf-8"))
        hasher.update(line[0].rstrip().encode("utf-8"))
        return hasher.hexdigest()

    def _to_uuid(self, chunk):
        if self._column_to_uuid is None:
            return
        col = self._column_to_uuid["original_column"]
        chunk[col] = pd.to_numeric(chunk[col]).apply(lambda x: f"{x:032x}")
        chunk.rename(columns={col: self._column_to_uuid["new_column"]}, inplace=True)

    def _finish_chunk(self, chunk: pd.DataFrame):
        self._to_uuid(chunk)
        self._q.put(chunk)

    def _run_w_cs(self):
        pd_reader = pd.read_csv(*self._args, **self._kwargs)
        kwargs = {
            k: v
            for k, v in self._kwargs.items()
            if k not in ["usecols", "parse_dates", "names", "dtypes"]
        }
        kwargs["delimiter"] = chr(15)  # set a dummy delimiter, so we basically read the whole line
        pd_summer = pd.read_csv(*self._args, **kwargs)
        for cs_chunk, df_chunk in zip(pd_summer, pd_reader):
            self.logger.debug(f"Read into queue: {df_chunk.index.max() + 1}")
            df_chunk["checksum"] = cs_chunk.apply(self._checksum_line, axis=1)
            self._finish_chunk(df_chunk)
        self._q.put(None)
