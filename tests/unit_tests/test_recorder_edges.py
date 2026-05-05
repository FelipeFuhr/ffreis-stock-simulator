from __future__ import annotations

from stock_simulator.recorder import InMemoryRecorder, NullRecorder
from stock_simulator.types import Action


def test_null_recorder_is_noop() -> None:
    recorder = NullRecorder()
    recorder.start_episode(seed=1)
    recorder.on_step(
        step=0,
        action=Action(side="hold"),
        fills=0,
        equity=100.0,
        leverage=0.0,
        price=10.0,
    )
    recorder.end_episode()
    assert recorder.replay() == ()


def test_inmemory_replay_filters_by_episode() -> None:
    recorder = InMemoryRecorder()

    recorder.start_episode(seed=1)
    recorder.on_step(
        step=0,
        action=Action(side="buy", units=1.0, order_type="market"),
        fills=1,
        equity=101.0,
        leverage=0.1,
        price=101.0,
    )
    recorder.end_episode()

    recorder.start_episode(seed=2)
    recorder.on_step(
        step=0,
        action=Action(side="sell", units=1.0, order_type="limit", limit_price=99.0),
        fills=1,
        equity=99.0,
        leverage=0.1,
        price=99.0,
    )

    all_rows = recorder.replay()
    ep0_rows = recorder.replay(episode_id=0)
    ep1_rows = recorder.replay(episode_id=1)

    assert len(all_rows) == 2
    assert len(ep0_rows) == 1
    assert len(ep1_rows) == 1
    assert ep0_rows[0].seed == 1
    assert ep1_rows[0].seed == 2
