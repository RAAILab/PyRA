# coding: utf-8
from typing import Dict, Optional
import datetime

import numpy as np
import pandas as pd
from pykrx import stock

from simulation.account import Account
from simulation.order import Order


# Data utility
def ticker_to_name(ticker: str) -> str:
    if ticker == 'cash':
        return '현금'
    else:
        return stock.get_market_ticker_name(ticker=ticker)


def get_lookback_fromdate(fromdate: str, lookback: int, freq: str) -> str:
    # freq에 따라 룩백 기간 포함된 예상 시작 날짜를 설정
    if freq == 'd':
        estimated_start_date = pd.to_datetime(fromdate) - pd.DateOffset(days=lookback*2)
    elif freq == 'm':
        estimated_start_date = pd.to_datetime(fromdate) - pd.DateOffset(months=lookback)
    elif freq == 'y':
        estimated_start_date = pd.to_datetime(fromdate) - pd.DateOffset(years=lookback)
    else:
        raise ValueError
    # 설정 기간(estimated_start_date ~ fromdate)의 KOSPI 데이터를 다운로드
    kospi = stock.get_index_ohlcv(fromdate=str(estimated_start_date.date()),
                                  todate=fromdate, ticker='1001', freq=freq)
    # 룩백 기간을 포함하는 정확한 시작 날짜를 반환
    return str(kospi.index[-lookback].date())


# Simulation utility
def order_target_amount(account: Account, dt: datetime.date,
                        ticker: str, target_amount: int) -> Optional[Order]:
    # 투자 포트폴리오의 각 자산 및 보유 수량
    positions = {asset_position.ticker: asset_position.position
                 for asset_position in account.portfolio.values()}
    # 자산의 보유 수량
    position = positions.get(ticker, 0)
    # 거래 수량 계산
    amount = target_amount - position
    if amount != 0:
        # 주문 객체 생성
        return Order(dt=dt, ticker=ticker, amount=amount)
    else:
        return None


def calculate_target_amount(account: Account, ticker: str,
                            target_percent: float, data: pd.DataFrame) -> int:
    assert ticker in data['ticker'].to_list()
    # 총 자산
    total_asset = account.total_asset
    # 자산의 현재 가격
    price = data.loc[data['ticker'] == ticker, 'close'].squeeze()
    # 목표 보유 수량 계산
    target_amount = int(np.fix(total_asset * target_percent / price))
    return target_amount


def order_target_percent(account: Account, dt: datetime.date, ticker: str,
                         target_percent: float, data: pd.DataFrame) -> Optional[Order]:
    # 목표 보유 수량 계산
    target_amount = calculate_target_amount(account=account, ticker=ticker,
                                            target_percent=target_percent, data=data)
    # 목표 수량에 따라 주문
    return order_target_amount(account=account, dt=dt, ticker=ticker,
                               target_amount=target_amount)


def rebalance(dt: datetime.date, data: pd.DataFrame, account: Account, weights: Dict):
    for asset_position in account.portfolio.values():
        if asset_position.ticker not in weights.keys():
            # 포트폴리오에 더 이상 포함되지 않는 기존 자산 매도
            order = order_target_percent(account=account, dt=dt,
                                         ticker=asset_position.ticker,
                                         target_percent=.0, data=data)
            # 주문 리스트에 생성된 주문 추가
            if order is not None:
                account.orders.append(order)

    for ticker, target_percent in weights.items():
        # 자산을 목표 편입비중으로 조정
        order = order_target_percent(account=account, dt=dt, ticker=ticker,
                                     target_percent=target_percent, data=data)
        if order is not None:
            # 주문 리스트에 생성된 주문 추가
            account.orders.append(order)
