from abc import ABCMeta, abstractmethod

from Backtest.event import FillEvent, OrderEvent
from Backtest.event import EventType

class ExecutionHandler(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute_order(self, event):
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, config, events, date_handler, compliance = None):
        self.events = events
        self.data_handler = date_handler
        self.compliance = compliance
        self.config = config



    def execute_order(self, event):
        if event.type == EventType.ORDER:
            ticker = event.ticker
            timestamp = self.data_handler.get_latest_bar_datetime(ticker)
            action = event.action
            quantity = event.quantity
            exchange = "ARCA"
            price = self.data_handler.get_latest_bar_value(ticker, "close")
            trade_mark = event.trade_mark
            if self.config['commission_ratio'] is not None:
                commission = self.config['commission_ratio']
            else:
                commission = None
            fill_event = FillEvent(timestamp, ticker, action, quantity, exchange, price, trade_mark, commission)
            self.events.put(fill_event)

            if self.compliance is not None:
                self.compliance.record_trade(fill_event)