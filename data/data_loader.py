import pandas as pd
import time
import exchange_calendars

from tqdm import tqdm
from pykrx import stock
from typing import Optional, List


class PykrxDataLoader:
    def __init__(self, fromdate: str, todate: str, market: str = "KOSPI"):
        self.fromdate = fromdate
        self.todate = todate
        self.market = market

    def get_ticker_list(self, date: Optional[str] = None):
        if date is None:
            date = self.todate
        return stock.get_market_ticker_list(date, market=self.market)

    def get_business_days(self):
        code = "XKRX"
        return exchange_calendars.get_calendar(code, start=self.fromdate, end=self.todate).sessions.strftime(
            "%Y-%m-%d").tolist()

    def load_market_cap_data(self, ticker_list: List, freq: str,
                             delay: float = 1):
        ticker_data_list = []
        for ticker in ticker_list:
            ticker_data = stock.get_market_cap(fromdate=self.fromdate,
                                               todate=self.todate,
                                               ticker=ticker, freq=freq)
            ticker_data = ticker_data.rename(
                columns={'시가총액': 'market_cap',
                         '거래량': 'volume',
                         '거래대금': 'trading_value',
                         '상장주식수': 'shares'}
            )
            ticker_data = ticker_data.assign(ticker=ticker)
            ticker_data.index.name = 'date'
            ticker_data_list.append(ticker_data)
            time.sleep(delay)
        data = pd.concat(ticker_data_list)

        return data

    # 기본 데이터 불러오기
    def load_fundamental_data(self, ticker_list: List, freq: str,
                              delay: float = 1):
        ticker_data_list = []
        for ticker in ticker_list:
            ticker_data = stock.get_market_fundamental(
                fromdate=self.fromdate, todate=self.todate,
                ticker=ticker, freq=freq)
            ticker_data = ticker_data.assign(ticker=ticker)
            ticker_data.index.name = 'date'
            ticker_data_list.append(ticker_data)
            time.sleep(delay)
        data = pd.concat(ticker_data_list)

        return data

    # 거래주체 데이터 불러오기
    def load_trader_data(self, ticker_list: List, freq: str,
                         delay: float = 1):
        ticker_data_list = []
        for ticker in ticker_list:
            ticker_data = stock.get_market_trading_value_by_date(
                fromdate=self.fromdate,
                todate=self.todate,
                ticker=ticker, freq=freq)
            ticker_data = ticker_data.rename(
                columns={'기관합계': 'institutional', '외국인합계': 'foreign',
                         '기타법인': 'other', '개인': 'individual', '전체': 'total'}
            )
            ticker_data = ticker_data.assign(ticker=ticker)
            ticker_data.index.name = 'date'
            ticker_data_list.append(ticker_data)
            time.sleep(delay)
        data = pd.concat(ticker_data_list)

        return data

    # 주가 데이터 불러오기
    def load_stock_data(self, ticker_list: List, freq: str, delay: float = 1):
        ticker_data_list = []
        for ticker in ticker_list:
            ticker_data = stock.get_market_ohlcv(fromdate=self.fromdate,
                                                 todate=self.todate,
                                                 ticker=ticker,
                                                 freq='d',
                                                 adjusted=True)
            ticker_data = ticker_data.rename(
                columns={'시가': 'open', '고가': 'high', '저가': 'low',
                         '종가': 'close', '거래량': 'volume',
                         '거래대금': 'trading_value', '등락률': 'change_pct'}
            )
            ticker_data = ticker_data.assign(ticker=ticker)
            ticker_data.index.name = 'date'
            ticker_data_list.append(ticker_data)
            time.sleep(delay)
        data = pd.concat(ticker_data_list)
        # 잠시 거래를 중단한 주식의 시가, 고가, 저가 보충
        data.loc[data.open == 0,
                 ['open', 'high', 'low']] = data.loc[data.open == 0, 'close']
        # 샘플링을 통해 일 데이터를 다른 주기 데이터로 변환
        if freq != 'd':
            rule = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                # 'trading_value': 'sum',
            }
            data = data.groupby('ticker').resample(freq).apply(rule).reset_index(level=0)
        data.__setattr__('frequency', freq)
        return data

    # etf 데이터 불러오기
    def load_etf_data(self, ticker_list: List, freq: str, delay: float = 1):
        ticker_data_list = []
        for ticker in ticker_list:
            ticker_data = stock.get_etf_ohlcv_by_date(fromdate=self.fromdate,
                                                      todate=self.todate,
                                                      ticker=ticker,
                                                      freq=freq)
            ticker_data = ticker_data.rename(
                columns={'NAV': 'nav', '시가': 'open', '고가': 'high', '저가': 'low',
                         '종가': 'close', '거래량': 'volume', '거래대금': 'trading_value',
                         '기초지수': 'benchmark'}
            )
            ticker_data = ticker_data.assign(ticker=ticker)
            ticker_data.index.name = 'date'
            ticker_data_list.append(ticker_data)
            time.sleep(delay)
        data = pd.concat(ticker_data_list)

        return data

    # 인덱스 데이터 불러오기
    def load_index_data(self, ticker_list: List, freq: str, delay: float = 1):
        ticker_data_list = []
        for ticker in ticker_list:
            ticker_data = stock.get_index_ohlcv(fromdate=self.fromdate,
                                                todate=self.todate,
                                                ticker=ticker,
                                                freq=freq)
            ticker_data = ticker_data.rename(
                columns={'날짜': 'date', '시가': 'open', '고가': 'high', '저가': 'low',
                         '종가': 'close', '거래량': 'volume', '거래대금': 'trading_value',
                         '등락률': 'change_pct', '상장시가총액': 'market_cap'}
            )
            ticker_data = ticker_data.assign(ticker=ticker)
            ticker_data.index.name = 'date'
            ticker_data_list.append(ticker_data)
            time.sleep(delay)
        data = pd.concat(ticker_data_list)

        return data

    # 해당 마켓의 전체 주가정보 불러오기
    def load_market_data(self, delay: float = 1):
        business_day_list = self.get_business_days()
        data = pd.DataFrame()
        for business_day in tqdm(business_day_list):
            df = stock.get_market_ohlcv(business_day, market=self.market)
            df.reset_index(inplace=True)
            df["date"] = business_day
            data = pd.concat([data, df])
            time.sleep(delay)
        data = data.rename(
            columns={'티커': 'ticker', '날짜': 'date', '시가': 'open', '고가': 'high',
                     '저가': 'low', '종가': 'close', '거래량': 'volume',
                     '거래대금': 'trading_value', '등락률': 'change_pct',
                     '상장시가총액': 'market_cap'}
        )

        return data
