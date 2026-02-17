from __future__ import annotations

from typing import Any, cast

import grpc
import numpy as np
from numpy.typing import NDArray

from stocksim_grpc import engine_pb2 as _engine_pb2
from stocksim_grpc import engine_pb2_grpc as _engine_pb2_grpc

engine_pb2: Any = cast(Any, _engine_pb2)
engine_pb2_grpc: Any = cast(Any, _engine_pb2_grpc)


class EngineGrpcClient:
    def __init__(self, *, target: str = "127.0.0.1:50051") -> None:
        self._channel = grpc.insecure_channel(target)
        self._stub = engine_pb2_grpc.EngineServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def ping(self) -> str:
        response = self._stub.Ping(engine_pb2.PingRequest())
        return str(response.status)

    def reset(self, seed: int | None = None) -> dict[str, float | int | bool]:
        request = engine_pb2.ResetRequest(has_seed=seed is not None, seed=int(seed or 0))
        response = self._stub.Reset(request)
        state = response.state
        return {
            "t": int(state.t),
            "cash": float(state.cash),
            "units": float(state.units),
            "equity": float(state.equity),
            "leverage": float(state.leverage),
            "open_orders": int(state.open_orders),
            "done": bool(state.done),
        }

    def observe(self) -> dict[str, object]:
        response = self._stub.Observe(engine_pb2.ObserveRequest())
        observation = response.observation
        return {
            "market_window_handle": {
                "start": int(observation.market_window_handle.start),
                "end": int(observation.market_window_handle.end),
                "t": int(observation.market_window_handle.t),
                "current_price": float(observation.market_window_handle.current_price),
            },
            "portfolio_vector": list(observation.portfolio_vector),
            "order_summary_vector": list(observation.order_summary_vector),
            "done": bool(observation.done),
        }

    def step_many(
        self, actions: NDArray[np.float64]
    ) -> tuple[list[dict[str, object]], NDArray[np.float64], NDArray[np.bool_]]:
        encoded_actions = []
        for row in actions:
            encoded_actions.append(
                engine_pb2.EncodedAction(
                    side_code=int(row[0]),
                    units=float(row[1]),
                    order_type_code=int(row[2]),
                    has_limit_price=not np.isnan(row[3]),
                    limit_price=0.0 if np.isnan(row[3]) else float(row[3]),
                )
            )
        response = self._stub.StepMany(engine_pb2.StepManyRequest(actions=encoded_actions))
        obs: list[dict[str, object]] = []
        for row in response.observations:
            obs.append(
                {
                    "market_window_handle": {
                        "start": int(row.market_window_handle.start),
                        "end": int(row.market_window_handle.end),
                        "t": int(row.market_window_handle.t),
                        "current_price": float(row.market_window_handle.current_price),
                    },
                    "portfolio_vector": list(row.portfolio_vector),
                    "order_summary_vector": list(row.order_summary_vector),
                    "done": bool(row.done),
                }
            )
        return (
            obs,
            np.asarray(response.rewards, dtype=np.float64),
            np.asarray(response.dones, dtype=np.bool_),
        )
