import pandas as pd
import numpy as np


def generate_bars(trading_datas, ticker, freq):
    '''
    for each ticker, organize transaction data into OHLC data
    '''

    trading_data = trading_datas[ticker]
    unit = str(freq) + "Min"

    trading_data["open"] = trading_data["last"]
    trading_data["high"] = trading_data["last"]
    trading_data["low"] = trading_data["last"]
    trading_data["close"] = trading_data["last"]
    trading_data['amount'] = trading_data['last'] * trading_data['volume']
    ohlc_data = trading_data.resample(unit).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum"
    })
    ohlc_data.loc[ohlc_data['volume'] == 0, 'volume'] = np.nan
    ohlc_data.loc[ohlc_data['amount'] == 0, 'amount'] = np.nan
    ohlc_data.fillna(method='backfill', inplace=True)

    return ohlc_data


    # ohlc_data = trading_data.groupby(trading_data.index).agg({
    #     "open": "first",
    #     "high": "max",
    #     "low": "min",
    #     "close": "last",
    #     "volume": "sum",
    #     "amount": "sum"
    # })