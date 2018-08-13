from abc import ABCMeta, abstractmethod

from Backtest.event import FillEvent, OrderEvent
from Backtest.event import EventType

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
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        '''
        Update the portfolio from the FILL event

        Parameters:
        event: class FillEvent
        '''
        raise NotImplementedError("Should implement update_fill()")

    @abstractmethod
    def update_timeindex(self, event):
        '''
        Update portfolio when now_time is changed
        '''
        raise NotImplementedError("Should implement update_timeindex()")




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
        self.data_handler = data_handler
        self.events = events
        self.start_date = start_date
        self.equity = equity
        self.tickers = self.data_handler.tickers

        self.all_positions = self._construct_all_positions()
        self.current_positions = self._construct_current_positions()

        self.all_holdings = self._construct_all_holdings()
        self.current_holdings = self._construct_current_holdings()

        self.current_tickers = []
        self.current_tickers_info = {}
        self.closed_positions = []

    def _construct_all_positions(self):
        '''
        Initialize self.all_positions
        '''
        d = dict([(ticker, 0) for ticker in self.tickers])
        d['datetime'] = self.start_date
        return [d]

    def _construct_current_positions(self):
        '''
        Initialize self.current_positions
        '''
        return dict([(ticker, 0) for ticker in self.tickers])

    def _construct_all_holdings(self):
        '''
        Initialize self.all_holdings
        '''
        d = dict([(ticker, 0.0) for ticker in self.tickers])
        d['datetime'] = self.start_date
        d['cash'] = self.equity
        d['commission'] = 0.0
        d['total'] = self.equity
        return [d]

    def _construct_current_holdings(self):
        '''
        Initialize self.current_holdings
        '''
        d = dict([(ticker, 0.0) for ticker in self.tickers])
        d['cash'] = self.equity
        d['commission'] = 0.0
        d['total'] = self.equity
        return d

    def update_timeindex(self, event):
        '''
        Update the self.all_positions and self.all_holdings
        from the MARKET event

        Parameters:
        event: class MarketEvent
        '''

        lasest_datetime = self.data_handler.get_latest_bar_datetime(event.ticker)

        # Update positions
        dposition = dict([(ticker, {}) for ticker in self.tickers])
        dposition['datetime'] = lasest_datetime
        for ticker in self.tickers:
            dposition[ticker] = self.current_positions[ticker]
        self.all_positions.append(dposition)

        # Update holdings
        dholding = dict([(ticker, 0.0) for ticker in self.tickers])
        dholding['datetime'] = lasest_datetime
        dholding['cash'] = self.current_holdings['cash']
        dholding['commission'] = self.current_holdings['commission']
        dholding['total'] = self.current_holdings['cash']
        for ticker in self.tickers:
            market_value = self.current_positions[ticker] * \
                self.data_handler.get_latest_bar_value(event.ticker, "close")
            dholding[ticker] = market_value
            dholding['total'] += market_value
        self.equity = dholding['total']
        self.all_holdings.append(dholding)

    def _update_positions_from_fill(self, event):
        '''
        Update self.current_positions from the FILL event

        Parameters:
        event: class FillEvent
        '''
        action_dir = {"LONG": 1, "SHORT": -1}
        self.current_positions[event.ticker] += action_dir.get(event.action, 0) * event.quantity

    def _update_holdings_from_fill(self, event):
        '''
        Update self.current_holdings from the FILL event

        Parameters:
        event: class FillEvent
        '''
        action_dir = {"LONG": 1, "SHORT": -1}
        action_dir.get(event.action, 0)
        action = action_dir.get(event.action, 0)
        event_cost = self.data_handler.get_latest_bar_value(event.ticker, "close")
        cost = action * event_cost * event.quantity
        self.current_holdings[event.ticker] += cost
        self.current_holdings['commission'] += event.commission
        self.current_holdings['cash'] -= (cost + event.commission)
        self.current_holdings['total'] -= (cost + event.commission)

    def _update_closed_postions_from_fill(self, event):
        '''
        Update self.current_tickers, self.current_tickers_info
        and self.closed_positions from the FILL event

        Parameters:
        event: class FillEvent
        '''
        if event.ticker not in self.current_tickers and event.action == "LONG":
            self.current_tickers.append(event.ticker)
            ticker_info = {
                "timestamp": event.timestamp,
                "ticker": event.ticker,
                "price": event.price,
                "quantity": event.quantity,
                "commission": event.commission
            }
            self.current_tickers_info[event.ticker] = ticker_info
        elif event.ticker in self.current_tickers and event.action == "SHORT":
            self.current_tickers.remove(event.ticker)
            closed_info = self.current_tickers_info.pop(event.ticker)
            closed = {
                "ticker": closed_info['ticker'],
                "quantity": closed_info['quantity'],
                "earning": event.price - closed_info['price'],
                "return": (event.price - closed_info['price']) / closed_info['price'],
                "timedelta": event.timestamp - closed_info['timestamp'],
                "commission_buy": closed_info['commission'],
                "commission_sell": event.commission
            }
            self.closed_positions.append(closed)
        elif event.ticker in self.current_tickers and event.action == "LONG":
            print("%s is already in position" % event.ticker)
        elif event.ticker not in self.current_tickers and event.action == "SHORT":
            print("%s is not in position" % event.ticker)

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
        ticker = event.ticker
        action = event.action
        trade_mark = event.trade_mark

        if event.suggested_quantity is None:
            quantity = 0
        else:
            quantity = event.suggested_quantity

        order_event = OrderEvent(ticker, action, quantity, trade_mark)
        return order_event

    def update_signal(self, event):
        '''
        Generate the ORDER event from the SIGNAL event

        Parameters:
        event: class SignalEvent
        '''
        if event.type == EventType.SIGNAL:
            order_event = self._generate_order(event)
            self.events.put(order_event)