# coding: utf-8
import numpy as np
import pandas as pd

annualization_factor = {
    'd': 252,
    'm': 12,
    'y': 1,
}


def cagr(returns: pd.Series, freq: str = 'd') -> float:
    if len(returns) < 1:
        return np.nan
    ann_factor = annualization_factor[freq]
    num_years = len(returns) / ann_factor
    cum_return = (returns + 1).prod()
    return cum_return ** (1 / num_years) - 1


def mdd(returns: pd.Series) -> float:
    if len(returns) < 1:
        return np.nan
    cum_returns = (returns + 1).cumprod()
    max_return = np.fmax.accumulate(cum_returns, axis=0)
    mdd = ((cum_returns - max_return) / max_return).min()
    return mdd


def sharpe_ratio(returns: pd.Series, risk_free: float = 0,
                 freq: str = 'd') -> float:
    if len(returns) < 2:
        return np.nan
    adjusted_returns = returns - risk_free
    ann_factor = annualization_factor[freq]
    sharpe_ratio = (adjusted_returns.mean() / adjusted_returns.std()
                    * np.sqrt(ann_factor))
    return sharpe_ratio


def sortino_ratio(returns: pd.Series, risk_free: float = 0,
                 freq: str = 'd') -> float:
    if len(returns) < 2:
        return np.nan
    adjusted_returns = returns - risk_free
    negative_returns = adjusted_returns[adjusted_returns < 0]
    downside_risk = np.sqrt(np.mean(negative_returns.pow(2)))
    ann_factor = annualization_factor[freq]
    if downside_risk == 0:
        sortino_ratio = np.NAN
    else:
        sortino_ratio = (adjusted_returns.mean() / downside_risk
                        * np.sqrt(ann_factor))
    return sortino_ratio
