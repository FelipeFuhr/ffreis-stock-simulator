from __future__ import annotations

import argparse
import importlib
import os
from concurrent import futures
from types import ModuleType
from typing import Any, cast

import grpc
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..config import GameConfig
from ..data import MarketData
from ..env import MarketEnv
from ..types import EnvState, Observation


def _load_grpc_module(module_name: str) -> ModuleType | None:
    """Load generated grpc module by name when present."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


engine_pb2 = _load_grpc_module("stocksim_grpc.engine_pb2")
engine_pb2_grpc = _load_grpc_module("stocksim_grpc.engine_pb2_grpc")


def _require_engine_pb2() -> ModuleType:
    """Return generated protobuf messages module or raise actionable error."""
    if engine_pb2 is None:
        raise RuntimeError(
            "gRPC protobuf messages are unavailable. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return engine_pb2


def _require_engine_pb2_grpc() -> ModuleType:
    """Return generated grpc stubs module or raise actionable error."""
    if engine_pb2_grpc is None:
        raise RuntimeError(
            "gRPC stubs are unavailable. Run ./scripts/generate_grpc_stubs.sh first."
        )
    return engine_pb2_grpc


def _build_synthetic_market_data(bars: int = 1024, seed: int = 1234) -> MarketData:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0001, scale=0.01, size=bars)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = rng.uniform(0.0002, 0.01, size=bars)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=bars, freq="h"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(10_000.0, 100_000.0, size=bars),
        }
    )
    return MarketData(frame)


def _load_market_data() -> MarketData:
    csv_path = os.getenv("MARKET_DATA_CSV", "").strip()
    if csv_path:
        return MarketData.from_csv(csv_path)
    bars = int(os.getenv("IPC_SYNTHETIC_BARS", "1024"))
    seed = int(os.getenv("IPC_SYNTHETIC_SEED", "1234"))
    return _build_synthetic_market_data(bars=bars, seed=seed)


def _observation_to_proto(observation: Observation) -> Any:
    pb2 = cast(Any, _require_engine_pb2())
    return pb2.Observation(
        market_window_handle=pb2.MarketWindowViewHandle(
            start=observation.market.start,
            end=observation.market.end,
            t=observation.market.t,
            current_price=observation.market.current_price,
        ),
        portfolio_vector=observation.portfolio_vector.tolist(),
        order_summary_vector=observation.order_summary_vector.tolist(),
        done=observation.done,
    )


def _state_to_proto(state: EnvState) -> Any:
    pb2 = cast(Any, _require_engine_pb2())
    return pb2.EnvState(
        t=state.t,
        cash=state.cash,
        units=state.units,
        equity=state.equity,
        leverage=state.leverage,
        open_orders=state.open_orders,
        done=state.done,
    )


class EngineGrpcService:
    def __init__(self, env: MarketEnv) -> None:
        self._env = env

    def Ping(  # noqa: N802
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        _ = (request, context)
        pb2 = cast(Any, _require_engine_pb2())
        return pb2.PingResponse(status="pong")

    def Reset(  # noqa: N802
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        _ = context
        pb2 = cast(Any, _require_engine_pb2())
        seed: int | None = int(request.seed) if request.has_seed else None
        state = self._env.reset(seed=seed)
        return pb2.ResetResponse(state=_state_to_proto(state))

    def Observe(  # noqa: N802
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        _ = (request, context)
        pb2 = cast(Any, _require_engine_pb2())
        observation = self._env.observe()
        return pb2.ObserveResponse(observation=_observation_to_proto(observation))

    def StepMany(  # noqa: N802
        self, request: Any, context: grpc.ServicerContext
    ) -> Any:
        pb2 = cast(Any, _require_engine_pb2())
        if len(request.actions) == 0:
            return pb2.StepManyResponse()

        rows: list[list[float]] = []
        for action in request.actions:
            limit_price = action.limit_price if action.has_limit_price else np.nan
            rows.append(
                [
                    float(action.side_code),
                    float(action.units),
                    float(action.order_type_code),
                    float(limit_price),
                ]
            )
        actions_matrix: NDArray[np.float64] = np.asarray(rows, dtype=np.float64)
        try:
            observations, rewards, dones = self._env.step_many(actions_matrix)
        except ValueError as exc:
            if context is None:
                raise
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            raise AssertionError("context.abort should raise") from exc

        observation_rows = []
        num_rows = int(rewards.shape[0])
        for i in range(num_rows):
            observation_rows.append(
                pb2.Observation(
                    market_window_handle=pb2.MarketWindowViewHandle(
                        start=int(observations["market_window_handle"][i, 0]),
                        end=int(observations["market_window_handle"][i, 1]),
                        t=int(observations["market_window_handle"][i, 2]),
                        current_price=float(observations["market_window_handle"][i, 3]),
                    ),
                    portfolio_vector=observations["portfolio_vector"][i].tolist(),
                    order_summary_vector=observations["order_summary_vector"][i].tolist(),
                    done=bool(dones[i]),
                )
            )
        return pb2.StepManyResponse(
            observations=observation_rows,
            rewards=rewards.tolist(),
            dones=dones.tolist(),
        )


def create_server(
    *,
    host: str,
    port: int,
    cfg: GameConfig | None = None,
    data: MarketData | None = None,
    max_workers: int = 16,
) -> grpc.Server:
    config = cfg if cfg is not None else GameConfig.load(yaml_path=os.getenv("STOCK_SIM_CONFIG_YAML"))
    market_data = data if data is not None else _load_market_data()
    env = MarketEnv(data=market_data, cfg=config)
    env.reset(seed=config.seed)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc = cast(Any, _require_engine_pb2_grpc())
    pb2_grpc.add_EngineServiceServicer_to_server(EngineGrpcService(env), server)
    server.add_insecure_port(f"{host}:{port}")
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="gRPC interface for stock simulator.")
    parser.add_argument("--host", default=os.getenv("GRPC_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GRPC_PORT", "50051")))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--use-numba", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = GameConfig(use_numba=args.use_numba, seed=args.seed)
    server = create_server(
        host=args.host,
        port=args.port,
        cfg=cfg,
        max_workers=args.max_workers,
    )
    server.start()
    print(f"stock simulator gRPC listening on {args.host}:{args.port}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
