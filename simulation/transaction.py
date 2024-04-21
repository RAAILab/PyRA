import datetime

import simulation.config as config
from simulation.order import OrderDirection


class Transaction(object):
    def __init__(self, id: str, dt: datetime.date, ticker: str, amount: int,
                 price: float, direction: OrderDirection,
                 commission_rate: float = config.commission_rate) -> None:
        self.id = id
        self.dt = dt
        self.ticker = ticker
        self.amount = amount
        self.price = price
        self.direction = direction
        self.commission_rate = commission_rate

        self.commission = (self.amount * self.price) * self.commission_rate
        self.settlement_value = -self.direction.value * (self.amount * self.price
                                                         ) - self.commission
