from abc import ABCMeta, abstractmethod

from Backtest.event import FillEvent, OrderEvent
from Backtest.event import EventType

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
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    def __init__(self, config, events, date_handler, compliance):
        '''
        Parameters:
        config: the list of settings showed in Backtest
        events: the event queue
        data_handler: class OHLCDataHandler
        compliance: class Compliance
        '''
        self.events = events
        self.data_handler = date_handler
        self.compliance = compliance
        self.config = config



    def execute_order(self, event):
        '''
        Execute the order:
            generate the FILL event
            and record the order             
        '''
        if event.type == EventType.ORDER:
            ticker = event.ticker
            timestamp = self.data_handler.get_latest_bar_datetime(ticker)
            action = event.action
            quantity = event.quantity
            exchange = self.config['exchange']
            price = self.data_handler.get_latest_bar_value(ticker, "close")
            trade_mark = event.trade_mark
            if self.config['commission_ratio'] is not None:
                commission = self.config['commission_ratio']
            else:
                commission = None
            fill_event = FillEvent(timestamp, ticker, action, quantity, exchange, price, trade_mark, commission)
            self.events.put(fill_event)

            self.compliance.record_trade(fill_event)