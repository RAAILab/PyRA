# coding: utf-8
import pandas as pd
from plotly import graph_objects as go, express as px


def plot_cumulative_return(returns: pd.Series, benchmark_returns: pd.Series,
                           strategy_name: str = 'My Strategy',
                           benchmark_name: str = 'KOSPI') -> None:
    # 포트폴리오의 누적 수익률 계산
    cum_returns = (returns + 1).cumprod() - 1
    # KOSPI의 누적 수익률 계산
    benchmark_cum_returns = (benchmark_returns + 1).cumprod() - 1

    # 그래프 객체
    fig = go.Figure()
    # 포트폴리오의 누적 수익률 곡선
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns,
                             name=strategy_name))
    # KOSPI의 누적 수익률 곡선
    fig.add_trace(go.Scatter(x=benchmark_cum_returns.index, y=benchmark_cum_returns,
                             name=benchmark_name, line = dict(dash='dot')))
    # 날짜 표시 형식
    fig.update_xaxes(tickformat='%Y-%m-%d')
    # 그래프 설정
    fig.update_layout(
        width=800,
        height=400,
        xaxis_title='날짜',
        yaxis_title='누적 수익률',
        legend_title_text='포트폴리오',
    )
    fig.show()


def plot_single_period_return(returns: pd.Series,
                              benchmark_returns: pd.Series,
                              strategy_name: str = 'My Strategy',
                              benchmark_name: str = 'KOSPI') -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=returns.index, y=returns,
                         name=strategy_name))
    fig.add_trace(go.Bar(x=benchmark_returns.index, y=benchmark_returns,
                         name=benchmark_name, marker_pattern_shape='/'))
    fig.update_xaxes(tickformat='%Y-%m-%d')
    fig.update_layout(
        width=800,
        height=400,
        xaxis_title='날짜',
        yaxis_title='수익률',
        legend_title_text='포트폴리오',
    )
    fig.show()


def plot_relative_single_period_return(returns: pd.Series,
                                       benchmark_returns: pd.Series) -> None:
    relative_returns = returns - benchmark_returns

    fig = go.Figure()
    fig.add_trace(go.Bar(x=relative_returns.index, y=relative_returns))
    fig.update_xaxes(tickformat='%Y-%m-%d')
    fig.update_layout(
        width=800,
        height=400,
        xaxis_title='날짜',
        yaxis_title='상대 수익률',
        legend_title_text=None,
    )
    fig.show()


def plot_cumulative_asset_profit(df_portfolio: pd.DataFrame) -> None:
    df_portfolio = df_portfolio.assign(
        profit=((df_portfolio['latest_price'] - df_portfolio['cost'])
                * df_portfolio['position']).fillna(0)
    )
    df_asset_profit = df_portfolio[['ticker', 'profit']].set_index(
        'ticker', append=True).unstack(level=-1, fill_value=0)['profit']
    df_asset_position = df_portfolio[['ticker', 'position']].set_index(
        'ticker', append=True).unstack(level=-1, fill_value=0)['position']
    df_asset_profit_change = df_asset_profit.diff()
    df_asset_profit_change[df_asset_position==0] = 0
    df_asset_cumulative_profit = df_asset_profit_change.cumsum()

    from plotly.validators.scatter.marker import SymbolValidator
    raw_symbols = SymbolValidator().values[2::12]

    fig = go.Figure()
    for idx, col in enumerate(df_asset_cumulative_profit.columns):
        if 'cash' in col:
            continue
        fig.add_trace(go.Scatter(x=df_asset_cumulative_profit.index,
                                 y=df_asset_cumulative_profit[col],
                                 name=col, mode='lines+markers',
                                 marker={'symbol': raw_symbols[idx]}))
    fig.update_xaxes(tickformat='%Y-%m-%d')
    fig.update_layout(
        width=800,
        height=400,
        xaxis_title='날짜',
        yaxis_title='누적 수익률',
        legend_title_text='종목코드',
    )
    fig.show()


def plot_asset_weight(df_portfolio: pd.DataFrame) -> None:
    # 현금의 보유수량을 1로 설정
    df_portfolio = df_portfolio.assign(
        position=df_portfolio['position'].fillna(1)
    )
    # 자산의 시가총액 계산
    df_portfolio = df_portfolio.assign(
        value=df_portfolio['latest_price'] * df_portfolio['position']
    )
    # 자산의 편입비중 계산
    df_portfolio = df_portfolio.assign(
        weight=df_portfolio['value'].groupby('date').transform(lambda x: x / x.sum())
    )

    fig = px.area(data_frame=df_portfolio, y='weight',
                  color='ticker', pattern_shape='ticker')
    fig.update_xaxes(tickformat='%Y-%m-%d')
    fig.update_layout(
        width=800,
        height=400,
        xaxis_title='날짜',
        yaxis_title='자산편입비중',
        legend_title_text='종목코드',
    )
    fig.show()
