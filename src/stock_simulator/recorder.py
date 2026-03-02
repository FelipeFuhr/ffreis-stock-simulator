from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from pandas import DataFrame as pd_DataFrame
from pandas import concat as pd_concat
from pandas import isna as pd_isna
from pandas import read_parquet as pd_read_parquet

from .types import Action

type ReplayScalar = int | float | str | None
type ReplayRow = dict[str, ReplayScalar]


@dataclass(frozen=True)
class RecordedStep:
    episode_id: int
    step: int
    seed: int
    side: str
    order_type: str
    units: float
    limit_price: float | None
    fills: int
    equity: float
    leverage: float
    price: float


class Recorder(ABC):
    @abstractmethod
    def start_episode(self, seed: int) -> None:
        """Initialize recorder state for a new episode."""

    @abstractmethod
    def on_step(
        self,
        *,
        step: int,
        action: Action,
        fills: int,
        equity: float,
        leverage: float,
        price: float,
    ) -> None:
        """Capture one simulation step."""

    @abstractmethod
    def end_episode(self) -> None:
        """Finalize current episode."""

    @abstractmethod
    def replay(self, episode_id: int | None = None) -> tuple[RecordedStep, ...]:
        """Return deterministic replay records."""


class NullRecorder(Recorder):
    def start_episode(self, seed: int) -> None:
        _ = seed

    def on_step(
        self,
        *,
        step: int,
        action: Action,
        fills: int,
        equity: float,
        leverage: float,
        price: float,
    ) -> None:
        _ = (step, action, fills, equity, leverage, price)

    def end_episode(self) -> None:
        return

    def replay(self, episode_id: int | None = None) -> tuple[RecordedStep, ...]:
        _ = episode_id
        return ()


class InMemoryRecorder(Recorder):
    def __init__(self) -> None:
        self._episode_id = -1
        self._seed = 0
        self._records: list[RecordedStep] = []

    def start_episode(self, seed: int) -> None:
        self._episode_id += 1
        self._seed = seed

    def on_step(
        self,
        *,
        step: int,
        action: Action,
        fills: int,
        equity: float,
        leverage: float,
        price: float,
    ) -> None:
        self._records.append(
            RecordedStep(
                episode_id=self._episode_id,
                step=step,
                seed=self._seed,
                side=action.side,
                order_type=action.order_type,
                units=float(action.units),
                limit_price=action.limit_price,
                fills=int(fills),
                equity=float(equity),
                leverage=float(leverage),
                price=float(price),
            )
        )

    def end_episode(self) -> None:
        return

    def replay(self, episode_id: int | None = None) -> tuple[RecordedStep, ...]:
        if episode_id is None:
            rows = self._records
        else:
            rows = [row for row in self._records if row.episode_id == episode_id]
        rows = sorted(rows, key=lambda r: (r.episode_id, r.step))
        return tuple(rows)


class ParquetRecorder(Recorder):
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._episode_id = -1
        self._seed = 0
        self._buffer: list[RecordedStep] = []

    def start_episode(self, seed: int) -> None:
        self._flush()
        self._episode_id += 1
        self._seed = seed

    def on_step(
        self,
        *,
        step: int,
        action: Action,
        fills: int,
        equity: float,
        leverage: float,
        price: float,
    ) -> None:
        self._buffer.append(
            RecordedStep(
                episode_id=self._episode_id,
                step=step,
                seed=self._seed,
                side=action.side,
                order_type=action.order_type,
                units=float(action.units),
                limit_price=action.limit_price,
                fills=int(fills),
                equity=float(equity),
                leverage=float(leverage),
                price=float(price),
            )
        )

    def end_episode(self) -> None:
        self._flush()

    def replay(self, episode_id: int | None = None) -> tuple[RecordedStep, ...]:
        self._flush()
        if not self._path.exists():
            return ()
        frame = pd_read_parquet(self._path)
        if frame.empty:
            return ()
        if episode_id is not None:
            frame = frame[frame["episode_id"] == episode_id]
        frame = frame.sort_values(["episode_id", "step"], kind="mergesort")
        rows = frame.to_dict(orient="records")
        replay_rows: list[RecordedStep] = []
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            typed_data = {str(key): value for key, value in data.items()}
            limit_raw = typed_data["limit_price"]
            replay_rows.append(
                RecordedStep(
                    episode_id=int(typed_data["episode_id"]),
                    step=int(typed_data["step"]),
                    seed=int(typed_data["seed"]),
                    side=str(typed_data["side"]),
                    order_type=str(typed_data["order_type"]),
                    units=float(typed_data["units"]),
                    limit_price=(None if pd_isna(limit_raw) else float(limit_raw)),
                    fills=int(typed_data["fills"]),
                    equity=float(typed_data["equity"]),
                    leverage=float(typed_data["leverage"]),
                    price=float(typed_data["price"]),
                )
            )
        return tuple(replay_rows)

    def _flush(self) -> None:
        if not self._buffer:
            return
        frame = pd_DataFrame([asdict(row) for row in self._buffer])
        if self._path.exists():
            previous = pd_read_parquet(self._path)
            frame = pd_concat([previous, frame], ignore_index=True)
        frame.to_parquet(self._path, index=False)
        self._buffer.clear()
