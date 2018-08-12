import queue

from Backtest.portfolio import PortfolioHandler
from Backtest.data import JSONDataHandler
from Backtest.execution import SimulatedExecutionHandler
from Backtest.performance import Performance
from Backtest.compliance import Compliance
from Backtest.event import EventType

class Backtest(object):
    def __init__(self, config, events_queue, strategy,
                 data_handler = None, portfolio_handler = None,
                 execution_handler = None, performance = None, compliance = None):
        self.config = config
        self.freq = config['freq']
        self.strategy = strategy
        self.tickers = config['tickers']
        self.equity = config['equity']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.events_queue = events_queue
        self.data_handler = data_handler
        self.portfolio_handler = portfolio_handler
        self.execution_handler = execution_handler
        self.performance = performance
        self.compliance = compliance
        self._config_backtest()
        self.cur_time = None

    def _config_backtest(self):
        if self.data_handler is None:
            self.data_handler = JSONDataHandler(
                self.config['csv_dir'], self.freq, self.events_queue, self.tickers,
                start_date = self.start_date, end_date = self.end_date
            )
        if self.portfolio_handler is None:
            self.portfolio_handler = PortfolioHandler(
                self.data_handler, self.events_queue,
                self.start_date, self.equity
            )
        if self.compliance is None:
            self.compliance = Compliance(
                self.config
            )
        if self.execution_handler is None:
            self.execution_handler = SimulatedExecutionHandler(
                self.events_queue, self.data_handler, self.compliance
            )
        if self.performance is None:
            self.performance = Performance(
                self.config, self.portfolio_handler, self.data_handler
            )

    def _continue_loop_condition(self):
        return self.data_handler.continue_backtest


    def _run_backtest(self):
        print("---------------------------------")
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
                            self.performance.update(event.timestamp)

                        if event.type == EventType.SIGNAL:
                            self.portfolio_handler.update_signal(event)

                        if event.type == EventType.ORDER:
                            self.execution_handler.execute_order(event)

                        if event.type == EventType.FILL:
                            self.portfolio_handler.update_fill(event)
        print("---------------------------------")
        print("Backtest complete.")
        print("---------------------------------")

    def _output_performance(self):
        results = self.performance.get_results()
        print("Sharpe Ratio: %0.10f" % results['sharpe'])
        print("Max Drawdown: %0.10f" % (results["max_drawdown"] * 100.0))
        print("Max Drawdown Duration: %d" % (results['max_drawdown_duration']))
        print("Total Returns: %0.10f" % (results['cum_returns'][-1] - 1))
        print("---------------------------------")
        print("Trades: %d" % results['trade_info']['trading_num'])
        print("Trade Winning: %s" % results['trade_info']['win_pct'])
        print("Average Trade: %s" % results['trade_info']['avg_trd_pct'])
        print("Average Win: %s" % results['trade_info']['avg_win_pct'])
        print("Average Loss: %s" % results['trade_info']['avg_loss_pct'])
        print("Best Trade: %s" % results['trade_info']['max_win_pct'])
        print("Worst Trade: %s" % results['trade_info']['max_loss_pct'])
        print("Worst Trade Date: %s" % results['trade_info']['max_loss_dt'])
        print("Avg Days in Trade: %s" % results['trade_info']['avg_dit'])
        print("---------------------------------")

        if self.config['is_plot'] or self.config['save_plot']:
            self.performance.plot_results(stats = results)
        return results


    def start_trading(self):
        self._run_backtest()
        results = self._output_performance()
        return results



