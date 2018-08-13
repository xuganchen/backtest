# DOCUMENT
This document is for _Backtest_.

## backtest.py

```python
class Backtest(object):
    '''
    the cerebrum of the backtesting system, running the event-loop calculation
    '''

    def __init__(self, config, events_queue, strategy, data_handler,
                 portfolio_handler = None, execution_handler = None, 
                 performance = None, compliance = None):
        '''
        Parameters:
        config: settings.
            {
            "csv_dir": input data path, 
            "out_dir": outputdata path,,
            "title": the title of strategy,
            "is_plot": whether plotting the result, True or False,
            "save_plot": whether saving the result, True or False,
            "save_tradelog": whether saving the trading log, True or False, 
            "start_date": pd.Timestamp("xxxx-xx-xxTxx:xx:xx", freq= str("freq") + "T"), 
                        strat datetime of backtesting
            "end_date": pd.Timestamp("xxxx-xx-xxTxx:xx:xx", freq= str("freq") + "T"), 
                        end datetime of backtesting
            "equity": initial funding,
            "freq": the frequency of backtesting,  a integer in minutes,
            "commission_ratio": the commission ratio of transaction, 
                                and the commission is "ratio * price * quantity"
            "exchange": the exchange
            "tickers": the list of trading digital currency.
            }

        events_queue: the event queue.
            queue.Queue()

        strategy: specific strategies adopted by users.
            class xxxxStrategy inherit from class Strategy

        data_handler: handling the work related to DATA, 
                      including inputting data, converting data, and updating tick data.
            class OHLCDataHandler

        portfolio_handler: handling situation of the positions 
                           and generates orders based on signals.
            class PortfolioHandler

        execution_handler: handling execution of orders. 
                           It represent a simulated order handling mechanism.
            class SimulatedExecutionHandler

        performance: calculating the backtest results and ploting the results.
            class Performance

        compliance: recording transaction information.
            class Compliance
        '''

    def _run_backtest(self):
        '''
        Main circulation department for event-driven backtest
        '''

    def _output_performance(self):
        '''
        Calculating the backtest results and ploting the results.

        return:
        results: a dict with all important results & stats.
        '''              
        return results

    def start_trading(self):        
        '''
        Start trading()
        '''
        self._run_backtest()
        results = self._output_performance()
        return results
```

## event.py

```python
'''
the standard event types for the following four events
'''
EventType = Enum("EventType", "MARKET TICK BAR SIGNAL ORDER FILL")
```

```python
class Event(object):
    '''
    Handling the work related to EVENT. 
    Event is base class providing an interface for all subsequent events.
    '''
```

```python
class MarketEvent(Event):
    """
    Handles the event of receiving a new market
    open-high-low-close-volume bar.
    """
    def __init__(self, ticker, timestamp, open, high, low, close, volume, freq):
        """
        Initialises the MarketEvent.

        Parameters:
        ticker: the ticker symbol
        timestamp: the timestamp of the bar
        open, high, close, low, volume: the information of the bar from data
        freq: the frequency in config, timedelta between every two bar
        """
```

```python
class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    def __init__(self, ticker, action, suggested_quantity, trade_mark):
        """
        Initialises the SignalEvent.

        Parameters:
        ticker: the ticker symbol
        action: "LONG"(for buy) or "SHORT"(for sell)
        suggested_quantity: Optional positively valued integer
            representing a suggested absolute quantity of units
            of an asset to transact in
        trade_mark: the mark when recorded into the log
                    determine by users in Strategy.generate_signals(),
                    such as "LONG", "SHORT", "ENMPY", 
                    "BUY", "SELL", "CLOSE" and etc.
        """
```

```python
class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    """
    def __init__(self, ticker, action, quantity, trade_mark):
        """
        Initialises the OrderEvent.

        Parameters:
        ticker: the ticker symbol
        action: "LONG"(for buy) or "SHORT"(for sell)
        quantity: the quantity to transact
        trade_mark: the mark when recorded into the log
                    determine by users in Strategy.generate_signals(),
                    such as "LONG", "SHORT", "ENMPY", 
                    "BUY", "SELL", "CLOSE" and etc.
        """
```

```python
class FillEvent(Event):
    """
    Encapsulates the notion of a filled order, as returned from a brokerage.
    """
    def __init__(self, timestamp, ticker, action, quantity, exchange, 
                price, trade_mark, commission_ratio):
        """
        Initialises the FillEvent object.

        timestamp: the timestamp of the bar
        ticker: the ticker symbol
        action: "LONG"(for buy) or "SHORT"(for sell)
        quantity: the quantity to transact
        exchange: he exchange where the order was filled.
        price: the price at which the trade was filled
        trade_mark: the mark when recorded into the log
                    determine by users in Strategy.generate_signals(),
                    such as "LONG", "SHORT", "ENMPY", 
                    "BUY", "SELL", "CLOSE" and etc.
        commission_ratio: the commission ratio of transaction, 
                          and the commission is "ratio * price * quantity"
        """
```

## portfolio.py
```python
class Portfolio(object):
    '''
    Handling situation of the positions 
    and generates orders based on signals.    
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def update_signal(self, event):
        '''
        Generate the ORDER event from the SIGNAL event

        Parameters:
        event: class SignalEvent
        '''

    @abstractmethod
    def update_fill(self, event):
        '''
        Update the portfolio from the FILL event

        Parameters:
        event: class FillEvent
        '''

    @abstractmethod
    def update_timeindex(self, event):
        '''
        Update portfolio when now_time is changed
        '''
```

```python
class PortfolioHandler(Portfolio):
    def __init__(self, data_handler, events, start_date, equity):
        '''
        Parameters:
        data_handler: class OHLCDataHandler
        events: the event queue
        start_date: strat datetime of backtesting
        equity: initial equity
        '''
        '''
        self.all_positions: positions situation for all past time
            a list of {"datetime": now, 
                        "ticker": self.current_positions[ticker] 
                        for each ticker}
        self.current_positions: positions situation now
            self.current_positions[ticker] = quantity of held ticker 


        self.all_holdings: holdings situation for all past time
            a list of {"datetime": now,
                        "ticker": self.current_holdings[ticker]
                        for each ticker}
        self.current_holdings: holdings situation now
            self.curent_holdings[ticker] = {"datetime": now,
                                            "cash": total cash now,
                                            "commission": total commissions, 
                                            "total": total equity,
                                            "ticker": market value 
                                            for each tickaer}


        self.current_tickers: currently held tickers
            a list of ticker name
        self.current_tickers_info: currently held tickers information
            self.current_tickers_info[ticker] = {
                "timestamp", "ticker", "price", "quantity", "commission"
            }
        self.closed_positions: already closed positions
            a list of {
                "ticker", "quantity",
                "earning": sell_price - buy_price,
                "return": (sell_price - buy_price) / buy_price,
                "timedelta": sell_timestamp - buy_timestamp,
                "commission_buy", "commission_sell"
            }
        '''

    def update_timeindex(self, event):
        '''
        Update the self.all_positions and self.all_holdings
        from the MARKET event

        Parameters:
        event: class MarketEvent
        '''

    def _update_positions_from_fill(self, event):
        '''
        Update self.current_positions from the FILL event

        Parameters:
        event: class FillEvent
        '''

    def _update_holdings_from_fill(self, event):
        '''
        Update self.current_holdings from the FILL event

        Parameters:
        event: class FillEvent
        '''

    def _update_closed_postions_from_fill(self, event):
        '''
        Update self.current_tickers, self.current_tickers_info
        and self.closed_positions from the FILL event

        Parameters:
        event: class FillEvent
        '''

    def update_fill(self, event):
        '''
        Update the portfolio from the FILL event

        Parameters:
        event: class FillEvent
        '''
        if event.type == EventType.FILL:
            self._update_positions_from_fill(event)
            self._update_holdings_from_fill(event)
            self._update_closed_postions_from_fill(event)

    def _generate_order(self, event):
        '''
        Generate the ORDER event from the SIGNAL event

        Parameters:
        event: class SignalEvent

        return:
        order_event: class OrderEvent
        '''

    def update_signal(self, event):
        '''
        Generate the ORDER event from the SIGNAL event

        Parameters:
        event: class SignalEvent
        '''
        if event.type == EventType.SIGNAL:
            order_event = self._generate_order(event)
            self.events.put(order_event)
```

## data.py
```python
class DataHandler(object):
    '''
    Handling the work related to DATA.
    DataHandler is base class providing an interface for all subsequent data handler.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, ticker):
        '''
        Get the latest bar information for ticker,
        '''

    @abstractmethod
    def get_latest_bars(self, ticker, N = 1):
        '''
        Get the latest N bar information for ticker
        '''

    @abstractmethod
    def get_latest_bar_datetime(self, ticker):
        '''
        Get the latest datetime
        '''

    @abstractmethod
    def get_latest_bar_value(self, ticker, val_type):
        '''
        Get the lastest bar's val_type
        '''

    @abstractmethod
    def get_latest_bars_values(self, ticker, val_type, N = 1):
        '''
        Get the lastest N bar's val_type
        '''

    @abstractmethod
    def update_bars(self):
        '''
        Update timeline for data 
        and generate MARKET event for each ticker
        '''
```

```python
class OHLCDataHandler(DataHandler):
    '''
    Handling the work related to DATA.
    The data format is "ticker timestamp open high low close"(OHLC)
    '''
    def __init__(self, csv_dir, freq, events_queue, tickers, start_date, 
                end_date, trading_data = None, ohlc_data = None):
        '''
        Parameters:
        csv_dir: input data path,
        freq: the frequency in config, timedelta between every two bar
        events_queue: the event queue
        tickers: the list of trading digital currency
        start_date: strat datetime of backtesting
        end_date: end datetime of backtesting
        trading_data: transaction data
            dict - trading_data[ticker] = df_ticker
                df_ticker = pd.DataFrame(index = "pd.timestamp", 
                                        columns = ["volume", "last"])
        ohlc_data: ohlc data
            dict - ohlc_data[ticker] = df_ticker
                df_ticker = pd.DataFrame(index = "pd.timestamp", 
                    columns = ["open", "high", "low", "close", "volume"])
        '''
        '''
        self.continue_backtest: the condition whether can continue backtesting,
                                determined by timeline
        self.trading_data = trading_data
        self.data = ohlc_data
        self.data_iter: for each ticker, the iterator of data DataFrame
        self.latest_data: for each ticker, past time data
        self.times: the time series of all time
        '''

    def generate_bars(self):
        '''
        for each ticker, organize transaction data into OHLC data
        '''

    def get_latest_bar(self, ticker):
        '''
        Get the latest bar information for ticker,
        return a N-row dict:
            {"timestamp",
            ["open", "high", "low", "close", "volume"]}
        '''

    def get_latest_bars(self, ticker, N = 1):
        '''
        Get the latest N bar information for ticker,
        return a 1-row dict:
            {"timestamp",
            ["open", "high", "low", "close", "volume"]}
        '''

    def get_latest_bar_datetime(self, ticker):
        '''
        Get the latest datetime
        return timestamp
        '''

    def get_latest_bar_value(self, ticker, val_type):
        '''
        Get the lastest bar's val_type
        return float
        
        Parameters:
        val_type: in ["open", "high", "low", "close", "volume"]
        '''

    def get_latest_bars_values(self, ticker, val_type, N = 1):
        '''
        Get the lastest N bar's val_type
        return np.array
        
        Parameters:
        val_type: in ["open", "high", "low", "close", "volume"]
        '''

    def update_bars(self):
        '''
        Update timeline for data 
        and generate MARKET event for each ticker

        If time is up, set continue_backtest = False
        and finish the backtest
        '''
```
## execution.py

```python
class ExecutionHandler(object):
    '''
    Handling execution of orders. 
    It represent a simulated order handling mechanism. 
    ExecutionHandler is base class providing an interface for all subsequent execution handler.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute_order(self, event):
        '''
        Execute the order:
            generate the FILL event
            and record the order             
        '''
```

```python
class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, config, events, date_handler, compliance):
        '''
        Parameters:
        config: the list of settings showed in Backtest
        events: the event queue
        data_handler: class OHLCDataHandler
        compliance: class Compliance
        '''

    def execute_order(self, event):
        '''
        Execute the order:
            generate the FILL event
            and record the order             
        '''
```
## strategy.py

```python
class Strategy(object):
    '''
    Handling all calculations on market data that generate trading signals.
    Strategy is base class providing an interface for all subsequent strategy.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, bars, events, suggested_quantity):
        '''
        Parameters:
        bars: class OHLCDataHandler
        events: the event queue
        suggested_quantity: Optional positively valued integer
            representing a suggested absolute quantity of units
            of an asset to transact in
        '''

    @abstractmethod
    def _calculate_initial_holdings(self):
        '''
        Calculate the status of all initial holdings, which is "EAMPY"
        '''

    @abstractmethod
    def generate_signals(self, event):
        '''
        Determine if there is a trading signal 
        and generate the SIGNAL event
        '''

    def generate_buy_signals(self, ticker, bar_date, str):
        '''
        If can_buy():
        print the information
        and generate LONG(buy) signal
        '''

    def generate_sell_signals(self, ticker, bar_date, str):
        '''
        If can_sell():
        print the information
        and generate SHORT(sell) signal
        '''
```

## compliance.py
```python
class AbstractCompliance(object):
    '''
    Recording transaction information.
    AbstractCompliance is base class providing an interface for all subsequent compliance.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def record_trade(self, fillevent):
        '''
        If ordering, record the trade into the log
        '''
```

```python
class Compliance(AbstractCompliance):
    def __init__(self, config):
        '''
        Parameters:
        config: the list of settings showed in Backtest
		
		save the basis information and log into fname, where fname is:
	        self.csv_fname = "tradelog_" + self.title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".csv"
	        fname = os.path.expanduser(os.path.join(self.out_dir, self.csv_fname))
        '''

    def record_trade(self, fillevent):
        '''
        If ordering, record the trade into the log

        Parameters:
        fillevent: class FillEvent
        '''
```
## performance.py
```python
from Backtest.plot_results import *

class Performance(object):
    '''
    Calculating the backtest results and ploting the results.
    '''
    def __init__(self, config, portfolio_handler, data_handler, periods = 365):
        '''
        Parameters:
        config: the list of settings showed in Backtest
        portfolio_handler: class PortfolioHandler
        data_handler: class OHLCDataHandler
        periods: trading day of the year
        '''

    def update(self, timestamp):
        '''
        Update performance's equity for every tick
        '''

    def _create_drawdown(self, cum_returns):
        '''
        Calculate drawdown
        '''

    def get_results(self):
        """
        Return a dict with all important results & stats.

        includings:
		    results['returns']
	        results['daily_returns']
	        results['equity']
	        results['rolling_sharpe']
	        results['cum_returns']
	        results['daily_cum_returns']
	        results['drawdown']
	        results['max_drawdown']
	        results['max_drawdown_duration']
	        results['sharpe']
	        results['positions']
            results['trade_info'] = {
                "trading_num": 'Trades Number'
                "win_pct": 'Trade Winning %'
                "avg_trd_pct": 'Average Trade %'
                "avg_win_pct": 'Average Win %'
                "avg_loss_pct": 'Average Loss %'
                "max_win_pct": 'Best Trade %'
                "max_loss_pct": 'Worst Trade %'
                "max_loss_dt": 'Worst Trade Date'
                "avg_dit": 'Avg Days in Trade'
            }
        """
        return results


    def plot_results(self, stats = None):
        '''
        Plot the results
        
        Parameters:
        stats = self.get_results()
        '''
```


#### plot_results.py
```python
def plot_equity(stats, ax=None, log_scale=False, **kwargs):
    '''
    Plots cumulative rolling returns
    '''

def plot_rolling_sharpe(stats, ax=None, **kwargs):
    '''
    Plots the curve of rolling Sharpe ratio.
    '''

def plot_drawdown(stats, ax=None, **kwargs):
    '''
    Plots the underwater curve
    '''

def plot_monthly_returns(stats, ax=None, **kwargs):
    '''
    Plots a heatmap of the monthly returns.
    '''

def plot_yearly_returns(stats, ax=None, **kwargs):
    '''
    Plots a barplot of returns by year.
    '''

def plot_txt_curve(stats, ax=None, periods = 365, **kwargs):
    """
    Outputs the statistics for the equity curve.
    """

def plot_txt_trade(stats, freq = 1, ax=None, **kwargs):
    '''
    Outputs the statistics for the trades.
    '''

def plot_txt_time(stats, ax=None, **kwargs):
    '''
    Outputs the statistics for various time frames.
    '''
```

