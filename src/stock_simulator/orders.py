from __future__ import annotations

from dataclasses import dataclass

from numpy import asarray as np_asarray
from numpy import float64 as np_float64
from numpy import int8 as np_int8
from numpy import logical_and as np_logical_and
from numpy import sum as np_sum
from numpy.typing import NDArray

from .core import CoreState


@dataclass(frozen=True)
class OrderSummary:
    open_orders: int
    buy_open_orders: int
    sell_open_orders: int

    def to_vector(self) -> NDArray[np_float64]:
        return np_asarray(
            [
                float(self.open_orders),
                float(self.buy_open_orders),
                float(self.sell_open_orders),
            ],
            dtype=np_float64,
        )


def summarize_orders(state: CoreState) -> OrderSummary:
    buy_open_orders = int(
        np_sum(
            np_logical_and(
                state.order_active == 1,
                state.order_side == np_int8(1),
            )
        )
    )
    sell_open_orders = int(
        np_sum(
            np_logical_and(
                state.order_active == 1,
                state.order_side == np_int8(-1),
            )
        )
    )
    return OrderSummary(
        open_orders=int(np_sum(state.order_active)),
        buy_open_orders=buy_open_orders,
        sell_open_orders=sell_open_orders,
    )
