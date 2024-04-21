from typing import Optional
from enum import Enum
import datetime
import uuid


class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOPMARKET = 3
    STOPLIMIT = 4


class OrderStatus(Enum):
    OPEN = 1
    FILLED = 2
    CANCELLED = 3


class OrderDirection(Enum):
    BUY = 1
    SELL = -1


class Order(object):
    def __init__(self, dt: datetime.date, ticker: str, amount: int,
                 type: Optional[OrderType] = OrderType.MARKET,
                 limit: Optional[float] = None, stop: Optional[float] = None,
                 id: Optional[str] = None) -> None:
        self.id = id if id is not None else uuid.uuid4().hex
        self.dt = dt
        self.ticker = ticker
        self.amount = abs(amount)
        self.direction = OrderDirection.BUY if amount > 0 else OrderDirection.SELL
        self.type = type
        self.limit = limit
        self.stop = stop

        self.status: OrderStatus = OrderStatus.OPEN
        self.open_amount: int = self.amount
