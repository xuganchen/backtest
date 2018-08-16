from abc import ABCMeta, abstractmethod
import queue
from Backtest.event import SignalEvent

class Strategy(object):
    '''
    Handling all calculations on market data that generate trading signals.
    Strategy is base class providing an interface for all subsequent strategy.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, config, events, data_handler):
        '''
        Parameters:
        data_handler: class OHLCDataHandler
        events: the event queue
        suggested_quantity: Optional positively valued integer
            representing a suggested absolute quantity of units
            of an asset to transact in
        '''
        self.config = config
        self.data_handler = data_handler
        self.tickers = self.config['tickers']
        self.events = events
    #     self.holdinds = self._calculate_initial_holdings()
    #
    # def _calculate_initial_holdings(self):
    #     holdings = {}
    #     for s in self.tickers:
    #         holdings[s] = "EMPTY"
    #     return holdings

    @abstractmethod
    def _calculate_initial_holdings(self):
        '''
        Calculate the status of all initial holdings, which is "EAMPY"
        '''
        raise NotImplementedError("Should implement _calculate_initial_holdings()")

    @abstractmethod
    def generate_signals(self, event):
        '''
        Determine if there is a trading signal 
        and generate the SIGNAL event
        '''
        raise NotImplementedError("Should implement generate_signals()")

    def generate_buy_signals(self, ticker, bar_date, str):
        '''
        If can_buy():
        print the information
        and generate LONG(buy) signal
        '''
        # print("%s: %s, %s" % (ticker, str, bar_date))
        signal = SignalEvent(ticker, "LONG", str)
        self.events.put(signal)

    def generate_sell_signals(self, ticker, bar_date, str):
        '''
        If can_sell():
        print the information
        and generate SHORT(sell) signal
        '''
        # print("%s: %s, %s" % (ticker, str, bar_date))
        signal = SignalEvent(ticker, "SHORT", str)
        self.events.put(signal)