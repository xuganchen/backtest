# Backtest
This is event-driven backtesting simulation written in Python.

* _Backtest_: the code of this backtesting system
* _DOCUMENT_: the documents of _Backtest_, _BayesianOptimization_ and _hyperopt_
* _strategy_: the folder of strategies
    * xxxxStrategy.py: the specific strategy, you can run them directly
    * xxxxStrategy_BO.ipynb: parameter adjusted file corresponding to "xxxxStrategy.py" using _Bayesian Optimization_
    * xxxxStrategy_hyperopt.ipynb: parameter adjusted file corresponding to "xxxxStrategy.py" using TPE based on [_hyperopt_](https://github.com/hyperopt/hyperopt) package
* _result_: results of some strategies using Beyesian Optimization and TPE. And the HTML file from xxxxStrategy_BO.ipynb and xxxxStrategy_hyperopt.ipynb
* _BayesianOptimization_: the method of Bayesian Optimization, uesd to adjust parameters
* _Binance_: data that have been processed into "OHLC" format. (using Backtest.open_json_gz_files and Backtest.generate bars)
* _archive_: the code for parameter adjustment using grid search

* Data format:

    * **trading_data**: transaction data
        dict - trading_data[ticker] = df_ticker
                df_ticker = pd.DataFrame(index = "pd.timestamp", columns = ["volume", "last"])
    * **ohlc_data**: ohlc data
        dict - ohlc_data[ticker] = df_ticker
                df_ticker = pd.DataFrame(index = "pd.timestamp", columns = ["open", "high", "low", "close", "volume", "amount"])

## Overview
### Event-Driven
It is handled by running the event-loop calculations, which can be simply expressed as below by pseudo-code:

This system was originally started with reference to [QuantStart](https://www.quantstart.com/articles).

```python
while True:								# run the loop forever by each tick
    try:
        new_event  = get_new_event()	# get the latest event
    except event_queue.Empty:	
        break							# until no new event
    else:
        if new_event.type == "MARKET":	# if it is a MARKET event:
        is_generate_signal()			# determine if there is a trading signal
										# and generate the SIGNAL event
        if new_event.type == "SIGNAL":	# if it is a SIGNAL event:
        generate_order()				# generate the ORDER event
										
        if new_event.type == "ORDER":	# if it is a ORDER event:
        execute_order()					# execute the order, record the order
										# and generate the FILL event
        if new_event.type == "FILL":	# if it is a FILL event:
        update_portfolio()				# update portfolio after ordering
```

### the Struture of Backtesting System
* _Backtest_: the cerebrum of the backtesting system, running the event-loop calculation as above.

* _Event_: handling the work related to EVENT. It contains a type (such as "MARKET", "SIGNAL", "ORDER" and "FILL") that determines how it will be handled in event-loop. 

* _Portfolio_: handling situation of the positions and generates orders based on signals.

* _Data_: handling the work related to DATA, including inputting data, converting data, and updating tick data.

* _Execution_: handling execution of orders. It represent a simulated order handling mechanism.

* _Strategy_: handling all calculations on market data that generate trading signals, including a based strategy and specific strategies adopted by users themselves.

* _Compliance_: recording transaction information.

* _Performance_: calculating the backtest results and ploting the results.

## Run backtest

### Quick Start
Import the package:
```python
import sys
backtest_dir = 'C://backtest/backtest/'
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)
    
from Backtest import *
```

Firtstly, initialize settings:

```python
config = {
    "csv_dir": "C:/backtest/Binance",
    "out_dir": "C:/backtest/results/MACDStrategy",
    "title": "MACDStrategy",
    "is_plot": True,
    "save_plot": True,
    "save_tradelog": True,
    "start_date": pd.Timestamp("2018-04-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
    "end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
    "equity": 1.0,
    "freq": 60,  # min
    "commission_ratio": 0.001,
    "suggested_quantity": None,     # None or a value
    "max_quantity": None,           # None or a value, Maximum purchase quantity
    "min_quantity": None,           # None or a value, Minimum purchase quantity
    "min_handheld_cash": None,      # None or a value, Minimum handheld funds
    "exchange": "Binance",
    "tickers": ['BTCUSDT']
}
```

* "csv_dir": input data path, 
* "out_dir": outputdata path,,
* "title": the title of strategy,
* "is_plot": whether plotting the result, True or False,
* "save_plot": whether saving the result, True or False,
* "save_tradelog": whether saving the trading log, True or False, 
* "start_date": pd.Timestamp("xxxx-xx-xxTxx:xx:xx", freq= str("freq") + "T"), strat datetime of backtesting
* "end_date": pd.Timestamp("xxxx-xx-xxTxx:xx:xx", freq= str("freq") + "T"), end datetime of backtesting
* "equity": initial funding,
* "freq": the frequency of backtesting,  a integer in minutes,
* "commission_ratio": the commission ratio of transaction, and the commission equantion is "cash * (1 + ratio) = price * quantity"
* "suggested_quantity": None or a value.if None, buy using all cash; if a value, buy fixed quantity
* "max_quantity": None or a value. If a value, this is the maximum number of transactions
* "min_quantity": None or a value. If a value, this is the minimum number of transactions
* "min_handheld_cash": None or a value. If a value, this is the minimum handheld funds
* "exchange": the exchange
* "tickers": the list of trading digital currency.

Secondly, define event queues:

```python
events_queue = queue.Queue()
```

Thirdly, input data and generate datahandler class:

```python
trading_data = {}
for ticker in config['tickers']:
    trading_data[ticker] = open_trading_data_files(config['csv_dir'], ticker)
    
data_handler = OHLCDataHandler(
    config['csv_dir'], config['freq'], events_queue, config['tickers'],
    start_date=config['start_date'], end_date=config['end_date'], 
    trading_data = trading_data, ohlc_data = None
    )
```
* **trading_data**: transaction data
    dict - trading_data[ticker] = df_ticker
            df_ticker = pd.DataFrame(index = "pd.timestamp", columns = ["volume", "last"])

**or**

```python
ohlc_data = {}
for ticker in config['tickers']:
    ohlc_data[ticker] = open_ohlc_data_files(config['csv_dir'], ticker)
    
data_handler = OHLCDataHandler(
        config['csv_dir'], config['freq'], events_queue, config['tickers'],
        start_date=config['start_date'], end_date=config['end_date'], 
    	trading_data = None, ohlc_data = ohlc_data
    )
```
* **ohlc_data**: ohlc data
    dict - ohlc_data[ticker] = df_ticker
            df_ticker = pd.DataFrame(index = "pd.timestamp", columns = ["open", "high", "low", "close", "volume", "amount"])

Fourthly, generate Strategy class:

```python
strategy = MACDStrategy(data_handler, events_queue, suggested_quantity = 100,
					short_window = short_window, long_window = long_window)
```

Then, generate Backtest class:

```python
backtest = Backtest(config, events_queue, strategy, data_handler)
```

Finnally, run the backtesting:

```python
results = backtest.start_trading()
```

### Specific Strategies Adopted by Users

```python
class xxxxStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1):	# required parameters
        self.bars = bars									    # data-handler class
        self.symbol_list = self.bars.tickers				    # the list of tickers
        self.events = events								    # event queue
        self.suggested_quantity = suggested_quantity			# suggested quantity to buy
        self.holdinds = self._calculate_initial_holdings()		# holding situation of tickers

    def _calculate_initial_holdings(self):					    # Required function, initilize
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            if buy_condition() and self.holdinds[ticker] == "EMPTY":    # if can_buy and "EMPTY":
                self.generate_buy_signals(ticker, bar_date, "LONG")     # generate the buy signal
                self.holdinds[ticker] = "HOLD"                          # change the status of ticker
            elif sell_condition() and self.holdinds[ticker] == "HOLD":  # if can_sell and "HOLDING":
                self.generate_sell_signals(ticker, bar_date, "SHORT")   # generate the sell signal
                self.holdinds[ticker] = "EMPTY"                         # and change the status of ticker
```

For example: 

```python
class MACDStrategy(Strategy):
    def __init__(self, bars, events, suggested_quantity = 1,    # required parameters
                 short_window = 10, long_window = 40):          # other parameter
        self.bars = bars									    # data-handler class
        self.symbol_list = self.bars.tickers                    # event queue
        self.events = events								    # event queue
        self.suggested_quantity = suggested_quantity			# suggested quantity to buy
        self.holdinds = self._calculate_initial_holdings()		# holding situation of tickers

        self.short_window = short_window
        self.long_window = long_window

    def _calculate_initial_holdings(self):					    # Required function, initilize
        holdings = {}
        for s in self.symbol_list:
            holdings[s] = "EMPTY"
        return holdings

    def generate_signals(self, event):
        if event.type == EventType.MARKET:
            ticker = event.ticker
            bar_date = event.timestamp
            bars = self.bars.get_latest_bars_values(            # get the lastest long_window price
                ticker, "close", N=self.long_window
            )
            if bars is not None and bars != []:
                short_ma = np.mean(bars[-self.short_window:])   # calculate short_ma
                long_ma  = np.mean(bars[-self.long_window:])    # calculate long_ma

                if short_ma > long_ma and self.holdinds[ticker] == "EMPTY":     # if can_buy and "EMPTY":
                    self.generate_buy_signals(ticker, bar_date, "LONG")         # generate the buy signal
                    self.holdinds[ticker] = "HOLD"                              # change the status of ticker
                elif short_ma < long_ma and self.holdinds[ticker] == "HOLD":    # if can_sell and "HOLDING":
                    self.generate_sell_signals(ticker, bar_date, "SHORT")       # generate the sell signal
                    self.holdinds[ticker] = "EMPTY"                             # and change the status of ticker
```


### Adjusting Paramaters Using Bayesian Optimization

_BayesianOptimization_: [fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization)

>This is a constrained global optimization package built upon bayesian inference and gaussian process, that attempts to find the maximum value of an unknown function in as few iterations as possible. This technique is particularly suited for optimization of high cost functions, situations where the balance between exploration and exploitation is important.

And I add several features into it, including passing invariants and the type of variable.

For example:
```python
import numpy as np
import pandas as pd
import queue
import matplotlib.pyplot as plt
import sys
backtest_dir = 'C://backtest/backtest/'
if backtest_dir not in sys.path:
    sys.path.insert(0, backtest_dir)
    
from Backtest import *
from BayesianOptimization import *
from ADXStrategy import ADXStrategy
from Backtest.open_json_gz_files import open_json_gz_files
from Backtest.generate_bars import generate_bars

def run_backtest(config, trading_data, ohlc_data, window):
    window = int(window)
    config['title'] = "ADXStrategy" + "_" + str(window)
    print("---------------------------------")
    print(config['title'])
    print("---------------------------------")
    
    events_queue = queue.Queue()

    data_handler = OHLCDataHandler(
        config, events_queue,
        trading_data = trading_data, ohlc_data = ohlc_data
    )
    strategy = ADXStrategy(config, events_queue, data_handler,
                           window = window)

    backtest = Backtest(config, events_queue, strategy,
                        data_handler= data_handler)

    results = backtest.start_trading()
    return (results['cum_returns'][-1] - 1)

config = {
    "csv_dir": "C:/backtest/Binance",
    "out_dir": "C:/backtest/results/ADXStrategy",
    "title": "ADXStrategy",
    "is_plot": False,
    "save_plot": False,
    "save_tradelog": False,
    "start_date": pd.Timestamp("2017-07-01T00:0:00", freq = "60" + "T"),    # str(freq) + "T"
    "end_date": pd.Timestamp("2018-09-01T00:00:00", freq = "60" + "T"),
    "equity": 1.0,
    "freq": 60,      # min
    "commission_ratio": 0.001,
    "suggested_quantity": None,     # None or a value
    "max_quantity": None,           # None or a value, Maximum purchase quantity
    "min_quantity": None,           # None or a value, Minimum purchase quantity
    "min_handheld_cash": None,      # None or a value, Minimum handheld funds
    "exchange": "Binance",
    "tickers": ['BTCUSDT']
}

ohlc_data = {}
for ticker in config['tickers']:
    ohlc_data[ticker] = pd.read_hdf(config['csv_dir'] + '\\' + ticker +'_OHLC_60min.h5', key=ticker)

trading_data = None
```

```python
gp_params = {"alpha": 1e-5}
BO = BayesianOptimization(
    run_backtest,
    {'window': (1, 240)},
    is_int = [1], 
    invariant = {
        'config': config,
        'trading_data': trading_data,
        'ohlc_data': ohlc_data
    },
    random_state = 1
)
BO.explore({
    'window': np.arange(1, 240, 20)
    },
    eager=True)
BO.maximize(init_points=0, n_iter=10, acq='ucb', kappa=5, **gp_params)
print(BO.res['max'])
```
More usage seeing example and documents.

### Adjusting Paramaters Using  TPE

_hyperopt_: [hyperopt/hyperopt](https://github.com/hyperopt/hyperopt)

>Hyperopt: Distributed Asynchronous Hyper-parameter Optimization

>Hyperopt is a Python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.

the document of the package is [http://hyperopt.github.io/hyperopt](http://hyperopt.github.io/hyperopt)

and the using of the function TPE is [https://github.com/hyperopt/hyperopt/wiki/FMin](https://github.com/hyperopt/hyperopt/wiki/FMin)

#### Installation
```python
pip install hyperopt
```

#### Simplest Case
```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def objective(x):
    return {'loss': x ** 2, 'status': STATUS_OK }

space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)
print(best)
```

## Class and Function Explanation

See _DOCUMENT for Backtest.md_, _DOCUMENT for BayesianOptimization.md_ and _DOCUMENT for hyperopt.md_

