# Backtest
This is event-driven backtesting simulation written in Python.

* _Backtest_: the code of this backtesting system
* xxxxStrategy.py: the specific strategy, you can run them directly
* xxxxStrategy_test.py: parameter adjusted file corresponding to "xxxxStrategy.py"
* _result_: results of some strategies
* _Binance_: data that have been processed into "OHLC" format. (using Backtest.open_json_gz_files and Backtest.generate bars)

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

Firtstly, initialize settings:

```python
config = {
	"csv_dir": "C:/backtest/trades_Bitfinex_folder",
	"out_dir": "C:/backtest/backtest/results/MACDStrategy",
	"title": "MACDStrategy",
	"is_plot": True,
	"save_plot": True,
	"save_tradelog": True,
	"start_date": pd.Timestamp("2017-01-01T00:0:00", freq="60" + "T"),  # str(freq) + "T"
	"end_date": pd.Timestamp("2018-09-01T00:00:00", freq="60" + "T"),
	"equity": 100000.0,
	"freq": 60,  # min
	"commission_ratio": 0.001,
    "exchange": "Binance", 
	"tickers": ['ETHUSD', 'BCHUSD', 'BCHBTC', 'BCHETH', 'EOSBTC']
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
* "commission_ratio": the commission ratio of transaction, and the commission is "ratio * price * quantity"
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



## Class and Function Explanation

See _DOCUMENT.md_

