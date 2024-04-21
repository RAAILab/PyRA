from simulation.transaction import Transaction


class AssetPosition(object):
    def __init__(self, ticker: str, position: int, latest_price: float, cost: float):
        self.ticker = ticker
        self.position = position
        self.latest_price = latest_price
        self.cost = cost

        self.total_settlement_value = (-1.0) * self.position * self.cost

    def update(self, transaction: Transaction):
        self.total_settlement_value += transaction.settlement_value
        self.position += transaction.direction.value * transaction.amount
        self.cost = (-1.0) * self.total_settlement_value / self.position \
            if self.position != 0 else 0.0
