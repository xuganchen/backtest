from abc import ABCMeta, abstractmethod
from datetime import datetime
import os
import csv

class AbstractCompliance(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def record_trade(self, fillevent):
        raise NotImplementedError("Should implement record_trade()")


class Compliance(AbstractCompliance):
    def __init__(self, config):
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

                writer = csv.DictWriter(file, fieldnames = fieldnames)
                writer.writeheader()

    def record_trade(self, fillevent):
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
