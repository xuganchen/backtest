from enum import Enum

'''
the standard event types for the following four events
'''
EventType = Enum("EventType", "MARKET TICK BAR SIGNAL ORDER FILL")

class Event(object):
    '''
    Handling the work related to EVENT. 
    Event is base class providing an interface for all subsequent events.
    '''
    @property
    def typename(self):
        return self.type.name


class MarketEvent(Event):
    """
    Handles the event of receiving a new market
    open-high-low-close-volume bar.
    """
    def __init__(self, ticker, timestamp, open, high, low, close, volume, amount, freq):
        """
        Initialises the MarketEvent.

        Parameters:
        ticker: the ticker symbol
        timestamp: the timestamp of the bar
        open, high, close, low, volume: the information of the bar from data
        freq: the frequency in config, timedelta between every two bar
        """
        self.type = EventType.MARKET
        self.ticker = ticker
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.close = close
        self.low = low
        self.volume = volume
        self.amount = amount
        self.freq = freq


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
        self.type = EventType.SIGNAL
        self.ticker = ticker
        self.action = action       
        self.suggested_quantity = suggested_quantity
        self.trade_mark = trade_mark


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
        self.type = EventType.ORDER
        self.ticker = ticker
        self.action  =action        # "LONG" or "SHORT"
        self.quantity = quantity
        self.trade_mark = trade_mark

    def print_order(self):
        print("Order: Ticker = %s, Action = %s, Quantity = %s" % (
            self.ticker, self.action, self.quantity
        ))

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
        self.type = EventType.FILL
        self.timestamp = timestamp
        self.ticker = ticker
        self.action = action
        self.quantity = quantity
        self.exchange = exchange
        self.price = price
        self.trade_mark = trade_mark
        if commission_ratio is None:
            self.commission = 0.001 * self.quantity * self.price
        else:
            self.commission = commission_ratio * self.quantity * self.price



