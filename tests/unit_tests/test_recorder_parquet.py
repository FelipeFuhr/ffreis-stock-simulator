from __future__ import annotations

from pathlib import Path

from pandas import DataFrame as pd_DataFrame
from pytest import MonkeyPatch

from stock_simulator import recorder as recorder_mod
from stock_simulator.recorder import ParquetRecorder
from stock_simulator.types import Action


def test_parquet_recorder_replay_when_file_missing(tmp_path: Path) -> None:
    recorder = ParquetRecorder(tmp_path / "missing.parquet")
    assert recorder.replay() == ()


def test_parquet_recorder_flush_and_replay_paths(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    store: dict[Path, pd_DataFrame] = {}

    def _fake_to_parquet(frame: pd_DataFrame, path: Path, index: bool = False) -> None:
        _ = index
        store[path] = frame.copy()
        path.write_text("stub", encoding="utf-8")

    def _fake_read_parquet(path: Path) -> pd_DataFrame:
        return store[path].copy()

    monkeypatch.setattr(recorder_mod.pd_DataFrame, "to_parquet", _fake_to_parquet)
    monkeypatch.setattr(recorder_mod, "pd_read_parquet", _fake_read_parquet)

    path = tmp_path / "replay.parquet"
    recorder = ParquetRecorder(path)

    recorder.start_episode(seed=10)
    recorder.on_step(
        step=0,
        action=Action(side="buy", units=1.0, order_type="market"),
        fills=1,
        equity=101.0,
        leverage=0.1,
        price=101.0,
    )
    recorder.end_episode()

    recorder.start_episode(seed=20)
    recorder.on_step(
        step=0,
        action=Action(side="sell", units=2.0, order_type="limit", limit_price=99.0),
        fills=1,
        equity=98.0,
        leverage=0.2,
        price=99.0,
    )
    recorder.end_episode()

    all_rows = recorder.replay()
    ep0_rows = recorder.replay(episode_id=0)
    ep1_rows = recorder.replay(episode_id=1)

    assert len(all_rows) == 2
    assert len(ep0_rows) == 1
    assert len(ep1_rows) == 1
    assert ep0_rows[0].seed == 10
    assert ep1_rows[0].seed == 20
    assert ep1_rows[0].limit_price == 99.0


def test_parquet_recorder_replay_empty_frame(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    path = tmp_path / "empty.parquet"
    path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(recorder_mod, "pd_read_parquet", lambda _path: pd_DataFrame())
    recorder = ParquetRecorder(path)
    assert recorder.replay() == ()
