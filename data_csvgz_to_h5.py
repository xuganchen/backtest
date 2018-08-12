import pandas as pd
from Backtest.open_gz_files import open_gz_files
import time

# csv_dir = 'F:\\Python\\backtest\\trades_Bitfinex_folder'
# ticker = 'ETHUSD'
#
# start = time.clock()
# df = open_gz_files(csv_dir, ticker)
# elapsed = (time.clock() - start)
# print("Time used:",elapsed)


csv_dir = 'F:\\Python\\backtest\\trades_Bitfinex_folder'
tickers = ['BCCBTC', 'BCCUSD', 'BCHBTC', 'BCHETH', 'BCHUSD',
            'ELFBTC', 'ELFETH', 'ELFUSD', 'EOSBTC', 'EOSETH',
            'EOSUSD', 'ETCBTC', 'ETCUSD', 'ETHBTC', 'ETHUSD',
            'IOSBTC', 'IOSETH', 'IOSUSD', 'LTCBTC', 'LTCUSD',
            'XRPBTC', 'XRPUSD']

for ticker in tickers:
    start = time.clock()
    h5 = pd.HDFStore(csv_dir + '\\' + ticker + '.h5', 'w')
    df = open_gz_files(csv_dir, ticker)
    h5[ticker] = df
    h5.close()
    elapsed = (time.clock() - start)
    print("Time for %s used:" % ticker, elapsed)
