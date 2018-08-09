
from __future__ import print_function
from enum import Enum


EventType = Enum("EventType", "MARKET TICK BAR SIGNAL ORDER FILL")

class Event(object):

    @property
    def typename(self):
        return self.type.name


class MarketEvent(Event):
    def __init__(self, ticker, timestamp, open, high, close, low, volume, freq):
        self.type = EventType.MARKET
        self.ticker = ticker
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.close = close
        self.low = low
        self.volume = volume
        self.freq = freq


class SignalEvent(Event):
    def __init__(self, ticker, action, suggested_quantity):
        self.type = EventType.SIGNAL
        self.ticker = ticker
        self.action = action        # "LONG" or "SHORT"
        self.suggested_quantity = suggested_quantity


class OrderEvent(Event):
    def __init__(self, ticker, action, quantity):
        self.type = EventType.ORDER
        self.ticker = ticker
        self.action  =action        # "LONG" or "SHORT"
        self.quantity = quantity

    def print_order(self):
        print("Order: Ticker = %s, Action = %s, Quantity = %s" % (
            self.ticker, self.action, self.quantity
        ))

class FillEvent(Event):
    def __init__(self, timestamp, ticker, action, quantity, exchange, price, commission = None):
        self.type = EventType.FILL
        self.timestamp = timestamp
        self.ticker = ticker
        self.action = action
        self.quantity = quantity
        self.exchange = exchange
        self.price = price
        if commission is None:
            self.commission = self.calculate_commission()
        else:
            self.commission = commission

    def calculate_commission(self):
        commission = 0.013 * self.quantity
        return commission


