from abc import ABCMeta, abstractmethod
from datetime import datetime
import os
import csv

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
        raise NotImplementedError("Should implement record_trade()")


class Compliance(AbstractCompliance):
    def __init__(self, config):
        '''
        Parameters:
        config: the list of settings showed in Backtest
        '''
        self.config = config
        self.out_dir = config['out_dir']
        self.title = config['title']
        self.config = config
        now = datetime.utcnow()
        self.csv_fname = "tradelog_" + self.title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".csv"

        if self.config['save_tradelog']:
            fieldnames = [
                "timestamp", "ticker", "action",
                "quantity", "exchange", "price", "commission"
            ]
            
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)

            fname = os.path.expanduser(os.path.join(self.out_dir, self.csv_fname))
            with open(fname, 'a') as file:
                writer = csv.writer(file)
                # record basis information
                writer.writerow([
                    self.config['title'],
                    "tickers", self.config['tickers'],
                    "initial equity", self.config['equity'],
                    "frequency", "%d min" % self.config['freq']
                ])
                writer.writerow([
                    "Start date", self.config['start_date'],
                    "End date", self.config['end_date']
                ])

                # record the field names
                writer = csv.DictWriter(file, fieldnames = fieldnames)
                writer.writeheader()

        # print("ticker: action, timestamp, price, quantity")

    def record_trade(self, fillevent):
        '''
        If ordering, record the trade into the log

        Parameters:
        fillevent: class FillEvent
        '''
        if self.config['save_tradelog']:
            fname = os.path.expanduser(os.path.join(self.out_dir, self.csv_fname))
            with open(fname, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([
                    fillevent.timestamp, fillevent.ticker,
                    fillevent.trade_mark, fillevent.quantity,
                    fillevent.exchange, fillevent.price,
                    fillevent.commission
                ])
        # print("%s: %s, %s, %.2f, %.6f" % (fillevent.ticker, fillevent.trade_mark,
        #                             fillevent.timestamp, fillevent.price, fillevent.quantity))

    def record_results(self, results):
        '''
        record all important results & stats.
        '''
        if self.config['save_tradelog']:
            fname = os.path.expanduser(os.path.join(self.out_dir, self.csv_fname))
            ratio = {
                "Sharpe Ratio": results['sharpe'],
                "Sortino Ratio": results['sortino'],
                "Max Drawdown": (results["max_drawdown"] * 100.0),
                "Max Drawdown Duration": (results['max_drawdown_duration']),
                "Total Returns": (results['cum_returns'][-1] - 1),
                "Annualized Returns": results['annual_return'],
                "Compound Annual Growth Rate": results['cagr']
            }

            trade = {
                "Trades": results['trade_info']['trading_num'],
                "Trade Winning": results['trade_info']['win_pct'],
                "Average Trade": results['trade_info']['avg_trd_pct'],
                "Average Win": results['trade_info']['avg_win_pct'],
                "Average Loss": results['trade_info']['avg_loss_pct'],
                "Best Trade": results['trade_info']['max_win_pct'],
                "Worst Trade": results['trade_info']['max_loss_pct'],
                "Worst Trade Date": results['trade_info']['max_loss_dt'],
                "Avg Days in Trade": results['trade_info']['avg_dit']
            }
            with open(fname, 'a') as file:
                headers = [k for k in ratio]
                writer = csv.DictWriter(file, fieldnames = headers)
                writer.writeheader()
                writer.writerow(ratio)

                headers = [k for k in trade]
                writer = csv.DictWriter(file, fieldnames = headers)
                writer.writeheader()
                writer.writerow(trade)