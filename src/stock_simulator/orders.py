from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core import CoreState


@dataclass(frozen=True)
class OrderSummary:
    open_orders: int
    buy_open_orders: int
    sell_open_orders: int

    def to_vector(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                float(self.open_orders),
                float(self.buy_open_orders),
                float(self.sell_open_orders),
            ],
            dtype=np.float64,
        )


def summarize_orders(state: CoreState) -> OrderSummary:
    buy_open_orders = int(
        np.sum(
            np.logical_and(
                state.order_active == 1,
                state.order_side == np.int8(1),
            )
        )
    )
    sell_open_orders = int(
        np.sum(
            np.logical_and(
                state.order_active == 1,
                state.order_side == np.int8(-1),
            )
        )
    )
    return OrderSummary(
        open_orders=int(np.sum(state.order_active)),
        buy_open_orders=buy_open_orders,
        sell_open_orders=sell_open_orders,
    )
