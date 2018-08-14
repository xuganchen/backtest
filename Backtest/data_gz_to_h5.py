import pandas as pd
import time

from open_gz_files import open_gz_files
from open_json_gz_files import open_json_gz_files
from generate_bars import generate_bars

# open csv.gz
# csv_dir = 'F:\\Python\\backtest\\trades_Bitfinex_folder'
# tickers = ['BCCBTC', 'BCCUSD', 'BCHBTC', 'BCHETH', 'BCHUSD',
#             'ELFBTC', 'ELFETH', 'ELFUSD', 'EOSBTC', 'EOSETH',
#             'EOSUSD', 'ETCBTC', 'ETCUSD', 'ETHBTC', 'ETHUSD',
#             'IOSBTC', 'IOSETH', 'IOSUSD', 'LTCBTC', 'LTCUSD',
#             'XRPBTC', 'XRPUSD']
#
# for ticker in tickers:
#     start = time.clock()
#     h5 = pd.HDFStore(csv_dir + '\\' + ticker + '.h5', 'w')
#     df = open_gz_files(csv_dir, ticker)
#     h5[ticker] = df
#     h5.close()
#     elapsed = (time.clock() - start)
#     print("Time for %s used:" % ticker, elapsed)


# open json.gz
# gz_dir = 'F:\\Python\\Binance'
# tickers = ['BTCUSDT']
# start_date = pd.Timestamp('2018-05-01')
# end_date = pd.Timestamp('2018-09-01')
# for ticker in tickers:
#     start = time.clock()
#     h5 = pd.HDFStore(gz_dir + '\\' + ticker + str(start_date.date())
#                      + 'to' + str(end_date.date()) + '.h5', 'w')
#     df = open_json_gz_files(gz_dir, ticker, start_date, end_date)
#     h5[ticker] = df
#     h5.close()
#     elapsed = (time.clock() - start)
#     print("Time for %s used:" % ticker, elapsed)


# generate bar
ticker = 'BTCUSDT'
csv_dir = 'C:/backtest/Binance'
trading_data = {}
trading_data[ticker] = pd.read_hdf(csv_dir + '\\' + ticker + '.h5', key=ticker)
df = generate_bars(trading_data, ticker, 1)
h5 = pd.HDFStore(csv_dir + '\\' + ticker + "_OHLC_1min.h5", 'w')
h5[ticker] = df
h5.close()

# ticker = 'BTCUSDT'
# csv_dir = 'C:/backtest/Binance'
# df = pd.read_hdf(csv_dir + '\\' + 'BTCUSDT_OHLC_1min.h5', key=ticker)
# print(df.head)