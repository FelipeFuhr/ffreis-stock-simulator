"""Server-side JSONL trace sink for ``STOCK_SIM_TRACE_JSONL`` (N10).

This is a debugging/verification aid, not a replay mechanism. It exists because
``step_many``'s ``include_trace`` flag is per-request and client-opted: the RL
agent's HTTP client never sets it, so a step's full trace (action, fill, and
post-step portfolio snapshot — see :class:`stock_simulator.types.StepTraceRow`)
is otherwise invisible to anyone who does not own the client code making
requests. Setting ``STOCK_SIM_TRACE_JSONL`` to a file path makes the server write
every step's trace row to that file as JSONL, regardless of what any individual
request asked for.

This is deliberately a separate mechanism from the structured episode-replay
port (:mod:`stock_simulator.recorder`, ``Recorder``/``RecordedStep``): that port
is wired into the singular ``MarketEnv.step`` call and captures a narrower,
replay-oriented schema (``episode_id, step, seed, side, order_type, units,
limit_price, fills, equity, leverage, price`` — no per-fill execution price).
It is never called from ``step_many`` (see the "Intentionally bypass
telemetry/recorder callbacks" comment in ``env.py``) — the RL agent's bulk HTTP
path deliberately skips it for headless-execution throughput. Reusing it here
would mean either wiring per-step callbacks into the hot bulk-execution loop
(the exact overhead ``step_many`` was written to avoid) or losing trace-level
fields (``StepTraceRow.exec_price`` from N9) the schema does not carry. A
dedicated, append-only, request-independent sink is the smaller and more
honest change.
"""

from __future__ import annotations

from json import dumps as json_dumps
from pathlib import Path
from typing import TextIO

from .types import StepTraceRow


class TraceJsonlWriter:
    """Append-only JSONL sink for :class:`StepTraceRow` trace rows.

    Opened once at server startup (see ``server.py::_load_engine``) when
    ``STOCK_SIM_TRACE_JSONL`` is set, then written to as steps happen. Uses
    line buffering (``buffering=1``) so each JSONL line reaches the OS promptly
    without an fsync per line — enough for a debugging aid; not a durability
    guarantee against the process crashing mid-line.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", buffering=1, encoding="utf-8")

    def write(self, row: StepTraceRow) -> None:
        """Append one trace row to the file as a single JSONL line."""
        self._file.write(json_dumps(row.to_serializable()) + "\n")

    def close(self) -> None:
        """Close the underlying file handle."""
        self._file.close()
