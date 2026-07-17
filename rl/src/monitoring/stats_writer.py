"""HDF5 stats persistence.

Separated from the callback so that the "what to record" logic stays in
`callbacks.py` and the "how to persist" logic lives here. This makes it
straightforward to add CSV / Parquet / W&B backends later — just write a
new writer with the same `append`/`flush` interface.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

# HDF5 ≥1.10 takes an advisory flock() on open. With the trainer writing while a
# read-only monitor (live_monitor / stats_reader / plot_run) polls the same file,
# the exclusive write-open is refused with BlockingIOError (errno 11, EAGAIN) and
# the whole run aborts. There is only ever one writer here and the monitors are
# read-only, so disabling the advisory lock is safe. Must be set before h5py opens
# any file (the HDF5 C lib reads it at open time) — set it before importing h5py.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np


class HDF5StatsWriter:
    """Append-only HDF5 writer for episode-level stats.

    Buffers rows in memory and flushes in batches. Each flush extends the
    on-disk datasets so a long run produces a single HDF5 file with one
    chunk per flush.
    """

    GROUP = "episodes"

    def __init__(self, path: Path, flush_every: int = 100) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = int(flush_every)
        self._buffer: list[Mapping[str, float]] = []

    # ------------------------------------------------------------------ API

    def append(self, row: Mapping[str, float]) -> None:
        """Buffer a single episode's record. Triggers a flush when full."""
        self._buffer.append(dict(row))
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def extend(self, rows: Iterable[Mapping[str, float]]) -> None:
        for row in rows:
            self.append(row)

    # Retry the open a few times before giving up: even with locking disabled this
    # guards against a transient OS-level contention so a stray hiccup never kills a
    # multi-hour run over a stats write.
    _OPEN_RETRIES = 5
    _OPEN_BACKOFF_S = 0.2

    def _open_append(self) -> h5py.File:
        last_err: OSError | None = None
        for attempt in range(self._OPEN_RETRIES):
            try:
                return h5py.File(self.path, "a")
            except (BlockingIOError, OSError) as err:
                last_err = err
                time.sleep(self._OPEN_BACKOFF_S * (attempt + 1))
        assert last_err is not None
        raise last_err

    def flush(self) -> None:
        """Write the buffer to disk. Idempotent on an empty buffer."""
        if not self._buffer:
            return

        keys = list(self._buffer[0].keys())
        arrays = {k: np.array([row[k] for row in self._buffer]) for k in keys}

        with self._open_append() as f:
            grp = f.require_group(self.GROUP)
            for k, arr in arrays.items():
                if k in grp:
                    ds = grp[k]
                    old_len = ds.shape[0]
                    ds.resize(old_len + len(arr), axis=0)
                    ds[old_len:] = arr
                else:
                    grp.create_dataset(
                        k,
                        data=arr,
                        maxshape=(None,),
                        compression="gzip",
                        chunks=True,
                    )
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
