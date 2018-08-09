from __future__ import print_function
from Backtest.event import EventType
import queue
from Backtest.portfolio import PortfolioHandler
from Backtest.data import JSONDataHandler
from Backtest.execution import SimulatedExecutionHandler
from Backtest.performance import Performance


class Backtest(object):
    def __init__(self, csv_dir, freq, strategy, tickers, equity, start_date, end_date, events_queue,
                end_test_time = None, data_handler = None, portfolio_handler = None,
                 execution_handler = None, performance = None):
        self.csv_dir = csv_dir
        self.freq = freq
        self.strategy = strategy
        self.tickers = tickers
        self.equity = equity
        self.start_date = start_date
        self.end_date = end_date
        self.events_queue = events_queue
        self.data_handler = data_handler
        self.portfolio_handler = portfolio_handler
        self.execution_handler = execution_handler
        self.performance = performance
        self._config_backtest()
        self.cur_time = None

    def _config_backtest(self):
        if self.data_handler is None:
            self.data_handler = JSONDataHandler(
                self.csv_dir, self.freq, self.events_queue, self.tickers,
                start_date = self.start_date, end_date = self.end_date
            )
        if self.portfolio_handler is None:
            self.portfolio_handler = PortfolioHandler(
                self.data_handler, self.events_queue,
                self.start_date, self.equity
            )
        if self.execution_handler is None:
            self.execution_handler = SimulatedExecutionHandler(
                self.events_queue, self.data_handler
            )
        if self.performance is None:
            self.performance = Performance(
                self.csv_dir, self.portfolio_handler, self.data_handler
            )

    def _continue_loop_condition(self):
        return self.data_handler.continue_backtest


    def _run_backtest(self):
        print("Running Backtest...")
        print("---------------------------------")
        i = 0
        while self._continue_loop_condition():
            i += 1
            # if i == 100:
            #     break
            # print(i)
            self.data_handler.update_bars()
            while True:
                try:
                    event = self.events_queue.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.type == EventType.MARKET:
                            self.strategy.generate_signals(event)
                            self.portfolio_handler.update_timeindex(event)
                            self.performance.update(event.timestamp, self.portfolio_handler)

                        if event.type == EventType.SIGNAL:
                            self.portfolio_handler.update_signal(event)

                        if event.type == EventType.ORDER:
                            self.execution_handler.execute_order(event)

                        if event.type == EventType.FILL:
                            self.portfolio_handler.update_fill(event)


    def _output_performance(self, out_dir = None, title = None, save_plot = False):
        results = self.performance.get_results()
        print("Sharpe Ratio: %0.10f" % results['sharpe'])
        print("Max Drawdown: %0.10f" % (results["max_drawdown"] * 100.0))
        print("Max Drawdown Duration: %d" % (results['max_drawdown_duration']))
        print("Total Returns: %0.10f" % (results['cum_returns'][-1] - 1))
        self.performance.plot_results(out_dir = out_dir, title = title, save_plot = save_plot)
        return results


    def start_trading(self, out_dir = None, title = None, save_plot = False):
        self._run_backtest()
        print("---------------------------------")
        print("Backtest complete.")
        print("---------------------------------")
        results = self._output_performance(out_dir = out_dir, title = title, save_plot = save_plot)
        return results



