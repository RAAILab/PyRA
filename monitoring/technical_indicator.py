import pandas as pd


def rsi(df: pd.DataFrame, window_length: int = 14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_length).mean()
    avg_loss = loss.rolling(window=window_length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 * rs / (1+rs)

    return rsi

def adr(df: pd.DataFrame, window_length: int = 20):
    ups = df.groupby("date")["change_pct"].apply(lambda x: (x>0).sum())
    downs = df.groupby("date")["change_pct"].apply(lambda x: (x<0).sum())

    sum_of_ups = ups.rolling(window=window_length).sum()
    sum_of_downs = downs.rolling(window=window_length).sum()
    adr = (sum_of_ups / sum_of_downs) * 100

    return adr

def macd(df: pd.DataFrame, window_length: tuple = (12, 26)):
    ma1 = df["close"].ewm(span=window_length[0]).mean()
    ma2 = df["close"].ewm(span=window_length[1]).mean()
    macd = ma1 - ma2 # MACD
    macds = macd.ewm(span=9).mean() # Signal
    macdh = macd - macds # Histogram

    return macdh