from typing import List, Dict, Optional, Tuple
import datetime
import pandas as pd

import simulation.config as config
from simulation.order import Order, OrderStatus, OrderDirection
from simulation.transaction import Transaction


class Broker(object):
    def __init__(self, slippage_rate: float = config.slippage_rate,
                 volume_limit_rate: float = config.volume_limit_rate):
        self.slippage_rate = slippage_rate
        self.volume_limit_rate = volume_limit_rate

    def calculate_slippage(self, data: Dict, order: Order) -> Tuple[float, int]:
        # 슬리피지를 포함한 거래 가격 계산
        price = data['open']
        simulated_impact = price * self.slippage_rate

        if order.direction == OrderDirection.BUY:
            impacted_price = price + simulated_impact
        else:
            impacted_price = price - simulated_impact

        # 거래가 가능한 수량 계산
        volume = data['volume']
        max_volume = volume * self.volume_limit_rate
        shares_to_fill = min(order.open_amount, max_volume)

        return impacted_price, shares_to_fill

    def process_order(self, dt: datetime.date, data: pd.DataFrame,
                      orders: Optional[List[Order]]) -> List[Transaction]:
        if orders is None:
            return []

        # 가격 데이터를 딕셔너리로 변환
        data = data.set_index('ticker').to_dict(orient='index')

        transactions = []
        for order in orders:
            if order.status == OrderStatus.OPEN:
                assert order.ticker in data.keys()
                # 슬리피지 계산
                price, amount = self.calculate_slippage(
                    data=data[order.ticker],
                    order=order
                )
                if amount != 0:
                    # 거래 객체 생성
                    transaction = Transaction(
                        id=order.id,
                        dt=dt,
                        ticker=order.ticker,
                        amount=amount,
                        price=price,
                        direction=order.direction,
                    )
                    transactions.append(transaction)
                    # 거래 객체의 상태와 미체결 수량 업데이트
                    if order.open_amount == transaction.amount:
                        order.status = OrderStatus.FILLED
                    order.open_amount -= transaction.amount

        return transactions
