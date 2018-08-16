import queue

from Backtest.portfolio import PortfolioHandler
from Backtest.data import OHLCDataHandler
from Backtest.execution import SimulatedExecutionHandler
from Backtest.performance import Performance
from Backtest.compliance import Compliance
from Backtest.event import EventType

class Backtest(object):
    '''
    the cerebrum of the backtesting system, running the event-loop calculation
    '''
    def __init__(self, config, events_queue, strategy, data_handler, 
                portfolio_handler = None, execution_handler = None, 
                performance = None, compliance = None):
        '''
        Parameters:
        config: settings.
            {
            "csv_dir": input data path, 
            "out_dir": outputdata path,,
            "title": the title of strategy,
            "is_plot": whether plotting the result, True or False,
            "save_plot": whether saving the result, True or False,
            "save_tradelog": whether saving the trading log, True or False, 
            "start_date": pd.Timestamp("xxxx-xx-xxTxx:xx:xx", freq= str("freq") + "T"), 
                        strat datetime of backtesting
            "end_date": pd.Timestamp("xxxx-xx-xxTxx:xx:xx", freq= str("freq") + "T"), 
                        end datetime of backtesting
            "equity": initial funding,
            "freq": the frequency of backtesting,  a integer in minutes,
            "commission_ratio": the commission ratio of transaction, 
                                and the commission is "ratio * price * quantity"
            "exchange": the exchange
            "tickers": the list of trading digital currency.
            }

        events_queue: the event queue.
            queue.Queue()

        strategy: specific strategies adopted by users.
            class xxxxStrategy inherit from class Strategy

        data_handler: handling the work related to DATA, 
                      including inputting data, converting data, and updating tick data.
            class OHLCDataHandler

        portfolio_handler: handling situation of the positions 
                           and generates orders based on signals.
            class PortfolioHandler

        execution_handler: handling execution of orders. 
                           It represent a simulated order handling mechanism.
            class SimulatedExecutionHandler

        performance: calculating the backtest results and ploting the results.
            class Performance

        compliance: recording transaction information.
            class Compliance
        '''
        self.config = config
        self.freq = config['freq']
        self.tickers = config['tickers']
        self.equity = config['equity']
        self.start_date = config['start_date']
        self.end_date = config['end_date']
        self.events_queue = events_queue
        self.strategy = strategy
        self.data_handler = data_handler
        self.portfolio_handler = portfolio_handler
        self.execution_handler = execution_handler
        self.performance = performance
        self.compliance = compliance
        self._config_backtest()

    def _config_backtest(self):
        '''
        Initialize the four class parameter, including 
            portfolio_handler, execution_handlerNone, performance and compliance.
        '''
        
        if self.portfolio_handler is None:
            self.portfolio_handler = PortfolioHandler(
                self.config, self.data_handler, self.events_queue
            )
        if self.compliance is None:
            self.compliance = Compliance(
                self.config
            )
        if self.execution_handler is None:
            self.execution_handler = SimulatedExecutionHandler(
                self.config, self.events_queue,
                self.data_handler, self.compliance
            )
        if self.performance is None:
            self.performance = Performance(
                self.config, self.portfolio_handler, self.data_handler
            )

    def _continue_loop_condition(self):
        '''
        Determine whether can continue to test back.
        '''
        return self.data_handler.continue_backtest


    def _run_backtest(self):
        '''
        Main circulation department for event-driven backtest
        '''
        print("---------------------------------")
        print("Running Backtest...")
        print("---------------------------------")
        while self._continue_loop_condition():          
            # update timeline for data and generate MARKET event for each ticker
            now_time = self.data_handler.update_bars()             
            # run the loop forever
            while True:                                 
                try:
                    # get the latest event
                    event = self.events_queue.get(False)    
                except queue.Empty:
                    # until no new event
                    break                                   
                else:
                    if event is not None:
                        if event.type == EventType.MARKET:                  
                            '''
                            if it is a MARKET event:
                            # determine if there is a trading signal 
                              and generate the SIGNAL event
                            '''
                            self.strategy.generate_signals(event)            

                        if event.type == EventType.SIGNAL:                  
                            '''
                            if it is a SIGNAL event:
                            # generate the ORDER event
                            '''
                            self.portfolio_handler.update_signal(event)     

                        if event.type == EventType.ORDER:                   
                            '''
                            if it is a ORDER event:
                            # execute the order, 
                              record the order 
                              and generate the FILL event
                            '''
                            self.execution_handler.execute_order(event)     

                        if event.type == EventType.FILL:                    
                            '''
                            # if it is a FILL event:
                            # update portfolio after ordering
                            '''
                            self.portfolio_handler.update_fill(event)  

            # update timeline for portfolio
            self.portfolio_handler.update_timeindex(now_time) 
            # update performance's equity for every tick
            self.performance.update(now_time)        

        print("---------------------------------")
        print("Backtest complete.")
        print("---------------------------------")

    def _output_performance(self):
        '''
        Calculating the backtest results and ploting the results.

        return:
        results: a dict with all important results & stats.
        '''
        # calculating the backtest results 
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

        # ploting the results.
        if self.config['is_plot'] or self.config['save_plot']:
            self.performance.plot_results(stats = results)                    
        return results


    def start_trading(self):        
        '''
        Start trading()
        '''
        self._run_backtest()
        results = self._output_performance()
        return results



