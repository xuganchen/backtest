import pandas as pd

def generate_bars(trading_datas, freq, tickers):
    datas = {}
    for ticker in self.tickers:
        trading_data = trading_datas[ticker]  
        unit = str(freq) + "Min"

        trading_data["open"] = trading_data["last"]
        trading_data["high"] = trading_data["last"]
        trading_data["low"] = trading_data["last"]
        trading_data["close"] = trading_data["last"]
        data = trading_data.resample(unit).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        data.loc[data['volume'] == 0, 'volume'] = np.nan
        data.fillna(method = 'backfill', inplace = True)
        datas[ticker] = data
    return datas