# coding: utf-8
from collections import defaultdict
from typing import Dict, Optional

import pandas as pd

from data.data_loader import PykrxDataLoader
from simulation.account import Account
from simulation.broker import Broker
from simulation.utility import rebalance


def date_adjust(index_df: pd.DataFrame, df: pd.DataFrame):
    new_index = downsample_df(index_df).index
    # 원래 인덱스에 새 인덱스 삽입
    df.index = df.index.map(
        lambda x: x.replace(day=new_index[
            (new_index.year == x.year) & (new_index.month == x.month)][0].day))

    return df


def downsample_df(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.index = pd.to_datetime(data.index)
    data['date'] = data.index
    data = data.resample('M').apply('last')

    return data.set_index(pd.DatetimeIndex(data.date)).drop(columns=['date'])


def calculate_momentum(ohlcv_data: pd.DataFrame,
                       lookback_period: int,
                       skip_period: int) -> pd.DataFrame:
    # 데이터 재구조화(OHLCV)
    data = ohlcv_data[['close', 'ticker']].reset_index().set_index(
        ['ticker', 'date']).unstack(level=0)['close']

    # 팩터 계산(모멘텀)
    momentum_data = data.shift(periods=skip_period).rolling(
        window=lookback_period).apply(lambda x: x[-1] / x[0] - 1)

    return momentum_data


def calculate_fundamental(ohlcv_data: pd.DataFrame,
                          market_cap_data: pd.DataFrame,
                          fundamental_data: pd.DataFrame,
                          strategy_name: str,
                          lookback_period: int) -> pd.DataFrame:
    # 룩백 길이 변환
    if ohlcv_data.frequency == 'd': lookback_period = int(lookback_period / 21)

    # 데이터 재구조화(OHLCV)
    data = ohlcv_data[['close', 'ticker']].reset_index().set_index(
        ['ticker', 'date']).unstack(level=0)['close']

    # 데이터 조정 및 재구조화(기본)
    fundamental_data = fundamental_adjuster(fundamental_data=fundamental_data,
                                            market_cap_data=market_cap_data)
    mapping = {'per': 'EPS', 'pbr': 'BPS', 'dividend': 'DPS'}
    target_fundamental = mapping.get(strategy_name)
    fundamental_data = fundamental_data[
        ['date', target_fundamental, 'ticker']].reset_index().set_index(
        ['ticker', 'date']).unstack(level=0)[target_fundamental]

    # 펀더멘탈 데이터 날짜 수정
    fundamental_data = date_adjust(index_df=data, df=fundamental_data)

    # 팩터 계산(PER, PBR, 배당)
    if strategy_name == "per":
        fundamental_data = fundamental_data.rolling(
            window=lookback_period).sum()
        fundamental_data = data / fundamental_data
    elif strategy_name == 'pbr':
        fundamental_data = data / fundamental_data
    elif strategy_name == 'dividend':
        fundamental_data = fundamental_data / data
    else:
        raise ValueError

    return fundamental_data


def fundamental_adjuster(fundamental_data: pd.DataFrame,
                         market_cap_data: pd.DataFrame) -> pd.DataFrame:
    # 현재 주식 수로 나누기
    market_cap_data['shares_div'] = market_cap_data['shares'].div(
        market_cap_data.groupby('ticker')['shares'].transform('last'))
    market_cap_data = market_cap_data[['shares_div', 'ticker']].reset_index()

    # 조정
    data = fundamental_data.reset_index().merge(
        market_cap_data, on=['date', 'ticker']).set_index(['date'])
    data['EPS'] = data['shares_div'] * data['EPS']
    data['BPS'] = data['shares_div'] * data['BPS']
    data['DPS'] = data['shares_div'] * data['DPS']

    # 주식 수 바뀐 당일 오류 탐색
    changed_row = data.copy()
    changed_row['previous'] = changed_row.shares_div.shift(1)
    changed_row = changed_row[changed_row.shares_div != changed_row.previous]
    changed_row = changed_row.reset_index().set_index(['ticker', 'date'])

    # 당일 오류 정정
    data = data.reset_index().set_index(['ticker', 'date'])
    for i, index in enumerate(data.index):
        if index in changed_row.index:
            data.loc[index] = data.iloc[i + 1]

    return data.reset_index()[['ticker', 'date', 'BPS', 'EPS', 'DPS']]


def calculate_small(ohlcv_data: pd.DataFrame,
                    market_cap_data: pd.DataFrame) -> pd.DataFrame:
    # 데이터 재구조화(OHLCV)
    data = ohlcv_data[['close', 'ticker']].reset_index().set_index(
        ['ticker', 'date']).unstack(level=0)['close']

    # 데이터 재구조화(주식 수)
    market_cap_data = \
        market_cap_data[['shares', 'ticker']].reset_index().set_index(
            ['ticker', 'date']).unstack(level=0)['shares']

    # 시장 데이터 날짜 수정
    market_cap_data = date_adjust(index_df=data, df=market_cap_data)

    # 팩터 계산(시가총액)
    market_cap_data = market_cap_data * data

    return market_cap_data


def calculate_lowvol(ohlcv_data: pd.DataFrame,
                     lookback_period: int) -> pd.DataFrame:
    # 데이터 재구조화
    data = ohlcv_data[['close', 'ticker']].reset_index().set_index(
        ['ticker', 'date']).unstack(level=0)['close']

    # 팩터 계산(표준편차)
    std_data = data.pct_change().rolling(lookback_period).std()

    return std_data


def calculate_trader(ohlcv_data: pd.DataFrame,
                     market_cap_data: pd.DataFrame,
                     trader_data: pd.DataFrame,
                     strategy_name: str,
                     lookback_period: Optional[int] = 1) -> pd.DataFrame:
    # 룩백 길이 변환
    if ohlcv_data.frequency == 'd': lookback_period = int(lookback_period / 21)

    # 데이터 재구조화(OHLCV)
    data = ohlcv_data[['close', 'ticker']].reset_index().set_index(
        ['ticker', 'date']).unstack(level=0)['close']

    # 데이터 재구조화(주식 수)
    market_cap_data = \
        market_cap_data[['shares', 'ticker']].reset_index().set_index(
            ['ticker', 'date']).unstack(level=0)['shares']

    # 데이터 재구조화(수급 주체)
    trader_data = \
        trader_data[[strategy_name, 'ticker']].reset_index().set_index(
            ['ticker', 'date']).unstack(level=0)[strategy_name]
    trader_data = trader_data.rolling(window=lookback_period).sum()

    # 시장 및 수급 주체 데이터 날짜 수정
    market_cap_data = date_adjust(index_df=data, df=market_cap_data)
    trader_data = date_adjust(index_df=data, df=trader_data)

    # 팩터 계산(수급 주체)
    market_cap_data = market_cap_data * data
    trader_data = trader_data / market_cap_data

    return trader_data


def get_factor_weight(factor_data: pd.DataFrame,
                      buying_ratio: float, strategy_name: str) -> Optional[
    Dict]:
    # 데이터 중 결측치가 있는지 확인함
    if factor_data.isnull().values.any():
        return None

    # 매수 주식 선정
    reverse = {'per', 'pbr', 'small', 'lovwol'}
    ratio = buying_ratio if strategy_name in reverse else 1 - buying_ratio
    top_quantile = factor_data.quantile(ratio)
    if strategy_name in reverse:
        stocks_to_buy = factor_data[factor_data <= top_quantile].index.to_list()
    else:
        stocks_to_buy = factor_data[factor_data >= top_quantile].index.to_list()

    # 주식 비율 할당
    weights = 1 / len(stocks_to_buy) if stocks_to_buy else 0
    portfolio = {ticker: weights if ticker in stocks_to_buy else 0.0 for ticker
                 in factor_data.index}

    return portfolio


def simulate_factor(ohlcv_data: pd.DataFrame,
                    market_cap_data: Optional[pd.DataFrame],
                    fundamental_data: Optional[pd.DataFrame],
                    trader_data: Optional[pd.DataFrame],
                    lookback_period: int,
                    skip_period: int,
                    strategy_name: str,
                    buying_ratio: float = 0.1) -> Account:
    # 계좌 및 브로커 선언
    account = Account(initial_cash=100000000)
    broker = Broker()

    # 팩터 계산
    if strategy_name == 'relative':
        factor_data = calculate_momentum(ohlcv_data=ohlcv_data,
                                         lookback_period=lookback_period,
                                         skip_period=skip_period, )
    elif strategy_name in {'per', 'pbr', 'dividend'}:
        factor_data = calculate_fundamental(ohlcv_data=ohlcv_data,
                                            market_cap_data=market_cap_data,
                                            fundamental_data=fundamental_data,
                                            lookback_period=lookback_period,
                                            strategy_name=strategy_name)
    elif strategy_name == 'small':
        factor_data = calculate_small(ohlcv_data=ohlcv_data,
                                      market_cap_data=market_cap_data)
    elif strategy_name in {'individual', 'institutional', 'foreign'}:
        factor_data = calculate_trader(ohlcv_data=ohlcv_data,
                                       market_cap_data=market_cap_data,
                                       trader_data=trader_data,
                                       lookback_period=lookback_period,
                                       strategy_name=strategy_name)
    elif strategy_name == 'lowvol':
        factor_data = calculate_lowvol(ohlcv_data=ohlcv_data,
                                       lookback_period=lookback_period)
    else:
        raise ValueError

    # 월별 리발란싱 날짜 추출
    month_end = downsample_df(ohlcv_data).index

    for date, ohlcv in ohlcv_data.groupby(['date']):
        print(date.date())
        # 주문 집행 및 계좌 갱신
        transactions = broker.process_order(dt=date, data=ohlcv,
                                            orders=account.orders)
        account.update_position(transactions=transactions)
        account.update_portfolio(dt=date, data=ohlcv)
        account.update_order()

        # 리발란싱 날짜가 아닐 경우 넘어가기
        if date not in month_end:
            continue

        # 팩터 전략을 이용하여 포트폴리오 구성
        factor_data_slice = factor_data.loc[date]
        weights = get_factor_weight(factor_data=factor_data_slice,
                                    buying_ratio=buying_ratio,
                                    strategy_name=strategy_name)

        print(f'Portfolio: {weights}')
        if weights is None:
            continue

        # 포트폴리오 비율 갱신
        account.update_weight(dt=date, weight=weights)

        # 주문 생성
        rebalance(dt=date, data=ohlcv, account=account, weights=weights)

    return account


if __name__ == '__main__':
    # 데이터 시작과 끝 날짜 정의
    fromdate = '2013-04-01'
    todate = '2021-12-30'

    # 투자할 종목 후보 정의
    ticker_list = ['000660', '005490', '051910', '006400', '005380', '000270',
                   '012330', '068270', '105560', '096770', '055550', '066570',
                   '047050', '032830', '015760', '086790', '000810', '033780',
                   '034730', '034020', '009150', '138040', '010130', '001570',
                   '010950', '024110', '030200', '051900', '009830', '086280',
                   '011170', '011070', '012450', '036570', '005830', '161390',
                   '034220', '004020', '032640', '097950', '000720', '006800',
                   '006260', '010620', '011780', '078930', '005940', '029780',
                   '128940', '035250', '016360', '021240', '010120', '052690',
                   '008770', '071050', '000990', '001450', '020150', '039490',
                   '111770', '000880', '004370', '036460', '007070', '138930',
                   '139480', ]

    # 데이터 불러오기
    data_loader = PykrxDataLoader(fromdate=fromdate, todate=todate,
                                  market="KOSPI")
    ohlcv_data_day = data_loader.load_stock_data(ticker_list=ticker_list,
                                                 freq='d', delay=1)
    fundamental_data = data_loader.load_fundamental_data(
        ticker_list=ticker_list, freq='m', delay=1)
    market_cap_data = data_loader.load_market_cap_data(ticker_list=ticker_list,
                                                       freq='m', delay=1)
    trader_data = data_loader.load_trader_data(ticker_list=ticker_list,
                                               freq='m', delay=1)

    setups = {
        'relative': {'ohlcv_data': ohlcv_data_day,
                     'market_cap_data': None,
                     'fundamental_data': None,
                     'trader_data': None,
                     'lookback_period': 3 * 21,
                     'skip_period': 1 * 21,
                     'strategy_name': 'relative',
                     'buying_ratio': 0.1,
                     },
        'per': {'ohlcv_data': ohlcv_data_day,
                'market_cap_data': market_cap_data,
                'fundamental_data': fundamental_data,
                'trader_data': None,
                'lookback_period': 12 * 21,
                'skip_period': 0,
                'strategy_name': 'per',
                'buying_ratio': 0.1,
                },
        'pbr': {'ohlcv_data': ohlcv_data_day,
                'market_cap_data': market_cap_data,
                'fundamental_data': fundamental_data,
                'trader_data': None,
                'lookback_period': 1 * 21,
                'skip_period': 0,
                'strategy_name': 'pbr',
                'buying_ratio': 0.1,
                },
        'dividend': {'ohlcv_data': ohlcv_data_day,
                     'market_cap_data': market_cap_data,
                     'fundamental_data': fundamental_data,
                     'trader_data': None,
                     'lookback_period': 1 * 21,
                     'skip_period': 0,
                     'strategy_name': 'dividend',
                     'buying_ratio': 0.1,
                     },
        'small': {'ohlcv_data': ohlcv_data_day,
                  'market_cap_data': market_cap_data,
                  'fundamental_data': None,
                  'trader_data': None,
                  'lookback_period': 1 * 21,
                  'skip_period': 0,
                  'strategy_name': 'small',
                  'buying_ratio': 0.1,
                  },
        'lowvol': {'ohlcv_data': ohlcv_data_day,
                   'market_cap_data': None,
                   'fundamental_data': None,
                   'trader_data': None,
                   'lookback_period': 60,
                   'skip_period': 0,
                   'strategy_name': 'lowvol',
                   'buying_ratio': 0.1,
                   },
        'individual': {'ohlcv_data': ohlcv_data_day,
                       'market_cap_data': market_cap_data,
                       'fundamental_data': None,
                       'trader_data': trader_data,
                       'lookback_period': 1 * 21,
                       'skip_period': 0,
                       'strategy_name': 'individual',
                       'buying_ratio': 0.1,
                       },
        'institutional': {'ohlcv_data': ohlcv_data_day,
                          'market_cap_data': market_cap_data,
                          'fundamental_data': None,
                          'trader_data': trader_data,
                          'lookback_period': 1 * 21,
                          'skip_period': 0,
                          'strategy_name': 'institutional',
                          'buying_ratio': 0.1,
                          },
        'foreign': {'ohlcv_data': ohlcv_data_day,
                    'market_cap_data': market_cap_data,
                    'fundamental_data': None,
                    'trader_data': trader_data,
                    'lookback_period': 1 * 21,
                    'skip_period': 0,
                    'strategy_name': 'foreign',
                    'buying_ratio': 0.1,
                    }
    }

    accounts = defaultdict(list)
    portfolios = []
    for name, setup in setups.items():
        result = simulate_factor(**dict(setup))
        accounts[name] = pd.DataFrame(result.account_history)[
            ['date', 'total_asset']].rename(columns={'total_asset': name})
        portfolio = pd.DataFrame(result.weight_history)
        portfolio = pd.melt(portfolio, id_vars=['date'], var_name='ticker',
                            value_name='weight',
                            value_vars=portfolio.columns[:-1])
        portfolio['factor'] = name
        portfolios.append(portfolio)
        print(f'strategy made {name}')

    factor_portfolio = pd.concat(portfolios).sort_values(
        by=['date', 'factor', 'ticker'])
    factor_asset = pd.concat(accounts, axis=1)
    factor_asset = factor_asset.droplevel(level=0, axis=1).T.drop_duplicates().T

    factor_portfolio.to_csv('factor_portfolio.csv')
    factor_asset.to_csv('factor_asset.csv')