from abc import ABCMeta, abstractmethod
import queue
from Backtest.event import SignalEvent

class Strategy(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, bars, events, suggested_quantity):
        self.bars = bars
        self.symbol_list = self.bars.tickers
        self.events = events
        self.suggested_quantity = suggested_quantity
        self.holdinds = self._calculate_initial_holdings()

    @abstractmethod
    def _calculate_initial_holdings(self):
        raise NotImplementedError("Should implement _calculate_initial_holdings()")

    @abstractmethod
    def generate_signals(self, event):
        raise NotImplementedError("Should implement generate_signals()")

    def generate_buy_signals(self, ticker, bar_date, str):
        print("%s: %s" % (str, bar_date))
        signal = SignalEvent(ticker, "LONG", self.suggested_quantity, str)
        self.events.put(signal)

    def generate_sell_signals(self, ticker, bar_date, str):
        print("%s: %s" % (str, bar_date))
        signal = SignalEvent(ticker, "SHORT", self.suggested_quantity, str)
        self.events.put(signal)