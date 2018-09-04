import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from Backtest.plot_results import *


class Performance(object):
    '''
    Calculating the backtest results and ploting the results.
    '''
    def __init__(self, config, portfolio_handler, data_handler, periods = 365):
        '''
        Parameters:
        config: the list of settings showed in Backtest
        portfolio_handler: class PortfolioHandler
        data_handler: class OHLCDataHandler
        periods: trading day of the year
        '''
        self.config = config
        self.portfolio_handler = portfolio_handler
        self.data_handler = data_handler
        self.equity = {}
        self.BNH_equity = {}
        self.periods = periods

    def update(self, timestamp):
        '''
        Update performance's equity for every tick
        '''
        self.equity[timestamp] = self.portfolio_handler.equity
        self.BNH_equity[timestamp] = self.portfolio_handler.BNH_equity

    def _create_drawdown(self, cum_returns):
        '''
        Calculate drawdown
        '''
        idx = cum_returns.index
        hwm = np.zeros(len(idx))

        for i in range(1, len(idx)):
            hwm[i] = max(hwm[i-1], cum_returns.iloc[i])

        dd = pd.DataFrame(index = idx)
        dd['Drawdown'] = (hwm - cum_returns) / hwm
        dd['Drawdown'].iloc[0] = 0.0
        dd['Duration'] = np.where(dd['Drawdown'] == 0, 0, 1)
        duration = max(sum(g) for k,g in groupby(dd['Duration']))
        return dd['Drawdown'], np.max(dd['Drawdown']), duration

    def _get_positions(self):
        positions = self.portfolio_handler.closed_positions
        if len(positions) == 0:
            return None
        else:
            return pd.DataFrame(positions)

    def get_results(self):
        """
        Return a dict with all important results & stats.
        
        includings:
            results['returns']
            results['daily_returns']
            results['equity']        
            results['tot_return']
            results['annual_return']
            results['cagr']
            results['rolling_sharpe']
            results['cum_returns']
            results['daily_cum_returns']
            results['drawdown']
            results['max_drawdown']
            results['max_drawdown_duration']
            results['sharpe']
            results['sortino']
            results['positions']
            results['trade_info'] = {
                "trading_num": 'Trades Number'
                "win_pct": 'Trade Winning %'
                "avg_trd_pct": 'Average Trade %'
                "avg_win_pct": 'Average Win %'
                "avg_loss_pct": 'Average Loss %'
                "max_win_pct": 'Best Trade %'
                "max_loss_pct": 'Worst Trade %'
                "max_loss_dt": 'Worst Trade Date'
                "avg_dit": 'Avg Days in Trade'
            }
        """
        # Equity
        res_equity = pd.Series(self.equity).sort_index()

        # Returns
        res_returns = res_equity.pct_change().fillna(0.0)
        res_daily_returns = res_equity.resample('D').last().pct_change().fillna(0.0)

        # Rolling Monthly Sharpe
        Monthly_periods = 30
        rolling = res_daily_returns.rolling(window = Monthly_periods)
        res_rolling_sharpe = np.sqrt(Monthly_periods) * (
            rolling.mean().dropna() / rolling.std().dropna()
        )

        # Cummulative Returns
        res_cum_returns = res_equity / self.config['equity']
        res_daily_cum_returns = res_cum_returns.resample("D").last()

        # totalreturn
        res_tot_return = res_cum_returns[-1] - 1.0

        # annualized rate of return
        times = res_equity.index
        years = (times[-1] - times[0]).total_seconds() / (self.periods * 24 * 60 * 60)
        res_annual_return = res_tot_return / years
        res_cagr = (res_cum_returns[-1] ** (1.0 / years)) - 1.0

        # Drawdown, max drawdown, max drawdown duration
        res_drawdown, res_max_dd, res_mdd_dur = self._create_drawdown(res_cum_returns)

        # Sharpe Ratio
        if np.std(res_daily_returns) == 0:
            res_sharpe = np.nan
        else:
            res_sharpe = np.sqrt(self.periods) * np.mean(res_daily_returns) / np.std(res_daily_returns)

        # sortino ratio
        if np.std(res_daily_returns[res_daily_returns < 0]) == 0:
            res_sortino = np.nan
        else:
            res_sortino = np.sqrt(self.periods) * (np.mean(res_daily_returns)) / np.std(res_daily_returns[res_daily_returns < 0])

        # BNH
        res_BNH_equity = pd.Series(self.BNH_equity).sort_index()
        res_BNG_returns = res_BNH_equity.pct_change().fillna(0.0)
        res_BNH_daily_returns = res_BNH_equity.resample("D").last().pct_change().fillna(0.0)
        res_BNH_cum_returns = res_BNH_equity / self.config['equity']
        IR_daily_returns = res_daily_returns - res_BNH_daily_returns
        if np.std(IR_daily_returns) == 0:
            res_IR = np.nan
        else:
            res_IR = np.sqrt(self.periods) * np.mean(IR_daily_returns) / np.std(IR_daily_returns)

        # rolling return 
        ## by Week
        res_rolling_return_week = res_equity.resample("W").apply(lambda x:x[-1] / x[0] - 1)
        res_rolling_return_week.index = res_rolling_return_week.index.date

        ## by Month
        res_rolling_return_month = res_equity.resample("M").apply(lambda x:x[-1] / x[0] - 1)
        res_rolling_return_month.index = [res_rolling_return_month.index.year, res_rolling_return_month.index.month] 

        ## by Year
        res_rolling_return_year = res_equity.resample("Y").apply(lambda x:x[-1] / x[0] - 1)
        res_rolling_return_year.index = res_rolling_return_year.index.year


        results = {}
        results['returns'] = res_returns
        results['daily_returns'] = res_daily_returns
        results['equity'] = res_equity
        results['tot_return'] = res_tot_return
        results['annual_return'] = res_annual_return
        results['cagr'] = res_cagr
        results['rolling_sharpe'] = res_rolling_sharpe
        results['cum_returns'] = res_cum_returns
        results['daily_cum_returns'] = res_daily_cum_returns
        results['drawdown'] = res_drawdown
        results['max_drawdown'] = res_max_dd
        results['max_drawdown_duration'] = res_mdd_dur
        results['sharpe'] = res_sharpe
        results['sortino'] = res_sortino
        results['IR'] = res_IR
        results['rolling_return_week'] = res_rolling_return_week
        results['rolling_return_month'] = res_rolling_return_month
        results['rolling_return_year'] = res_rolling_return_year
        results['BNH_equity'] = res_BNH_equity
        results['BNH_returns'] = res_BNG_returns
        results['BNH_cum_returns'] = res_BNH_cum_returns

        positions = self._get_positions()
        if positions is not None:
            results['positions'] = positions
            num_trades = positions.shape[0]
            win_pct = positions[positions["return"] > 0].shape[0] / float(num_trades)
            avg_trd_pct = np.mean(positions["return"])
            avg_win_pct = np.mean(positions[positions["return"] > 0]["return"])
            avg_loss_pct = np.mean(positions[positions["return"] <= 0]["return"])
            max_win_pct = np.max(positions["return"])
            max_loss_pct = np.min(positions["return"])
            max_loss = positions[positions["return"] == np.min(positions["return"])]
            max_loss_dt = np.mean(max_loss["timedelta"]).round(freq = "s")
            avg_dit = np.mean(positions["timedelta"]).round(freq = "s")
            results['trade_info'] = {
                "trading_num": num_trades,      # 'Trades'
                "win_pct": win_pct,             # 'Trade Winning %'
                "avg_trd_pct": avg_trd_pct,     # 'Average Trade %'
                "avg_win_pct": avg_win_pct,     # 'Average Win %'
                "avg_loss_pct": avg_loss_pct,   # 'Average Loss %'
                "max_win_pct": max_win_pct,     # 'Best Trade %'
                "max_loss_pct": max_loss_pct,   # 'Worst Trade %'
                "max_loss_dt": max_loss_dt,     # 'Worst Trade Date'
                "avg_dit": avg_dit              # 'Avg Days in Trade'
            }
        else:
            results['positions'] = None
            results['trade_info'] = {
                "trading_num": 0,               # 'Trades'
                "win_pct": np.nan,              # 'Trade Winning %'
                "avg_trd_pct": np.nan,          # 'Average Trade %'
                "avg_win_pct": np.nan,          # 'Average Win %'
                "avg_loss_pct": np.nan,         # 'Average Loss %'
                "max_win_pct": np.nan,          # 'Best Trade %'
                "max_loss_pct": np.nan,         # 'Worst Trade %'
                "max_loss_dt": 0,               # 'Worst Trade Date'
                "avg_dit": 0                    # 'Avg Days in Trade'
            }

        return results

    def plot_results(self, stats = None):
        '''
        Plot the results
        
        Parameters:
        stats = self.get_results()
        '''
        self.title = self.config['title']

        rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.995',
            'figure.facecolor': '0.97',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.labelweight': 'bold',
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        }
        sns.set_context(rc)
        sns.set_style("whitegrid")
        sns.set_palette("deep", desat=.6)
        vertical_sections = 7
        fig = plt.figure(figsize=(10, vertical_sections * 3.5))
        fig.suptitle(self.title, y=0.94, weight='bold')
        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.5)

        if stats is None:
            stats = self.get_results()

        ax_equity = plt.subplot(gs[:2, :])
        ax_sharpe = plt.subplot(gs[2, :])
        ax_drawdown = plt.subplot(gs[3, :])
        ax_week_returns = plt.subplot(gs[4, :])
        ax_monthly_returns = plt.subplot(gs[5, :2])
        ax_yearly_returns = plt.subplot(gs[5, 2])
        ax_txt_curve = plt.subplot(gs[6, 0])
        ax_txt_trade = plt.subplot(gs[6, 1])
        ax_txt_time = plt.subplot(gs[6, 2])

        plot_equity(stats, self.config, ax=ax_equity)
        plot_rolling_sharpe(stats, ax=ax_sharpe)
        plot_drawdown(stats, ax=ax_drawdown)
        plot_weekly_returns(stats, ax=ax_week_returns)
        plot_monthly_returns(stats, ax=ax_monthly_returns)
        plot_yearly_returns(stats, ax=ax_yearly_returns)
        plot_txt_curve(stats, ax=ax_txt_curve, periods = self.periods)
        plot_txt_trade(stats, ax=ax_txt_trade, freq = self.config['freq'])
        plot_txt_time(stats, ax=ax_txt_time)

        plt.show(block=False)
        if self.config is not None and self.config['save_plot'] == True:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title']
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')



    def plot_cum_returns(self, stats, log_scale=False, mid_time=None, savefig=False, **kwargs):
        '''
        Plots cumulative rolling returns

        Parameters:
        mid_time: the straight line
        stats = self.get_results()
        savefig = True or False
        log_scale = True or False
        '''
        if stats is None:
            stats = self.get_results()

        plt.figure(figsize=(12,5))
        plot_equity(stats, self.config, log_scale=log_scale, mid_time=mid_time, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "cum_returns"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_rolling_sharpe(self, stats, mid_time=None, savefig=False, **kwargs):
        '''
        Plots the curve of rolling Sharpe ratio.

        Parameters:
        mid_time: the straight line
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
            
        plt.figure(figsize=(12,5))
        plot_rolling_sharpe(stats, mid_time=mid_time, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "rolling_sharpe"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_drawdown(self, stats, mid_time=None, savefig=False, **kwargs):
        '''
        Plots the underwater curve

        Parameters:
        mid_time: the straight line
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
            
        plt.figure(figsize=(12,5))
        plot_drawdown(stats, mid_time=mid_time, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "drawdown"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_weekly_returns(self, stats, savefig=False, **kwargs):
        '''
        Plots a bar of the weekly returns.

        Parameters:
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
            
        plt.figure(figsize=(12,5))
        plot_weekly_returns(stats, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "weely_returns"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_monthly_returns(self, stats, savefig=False, **kwargs):
        '''
        Plots a heatmap of the monthly returns.

        Parameters:
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
            
        plt.figure(figsize=(12,5))
        plot_monthly_returns(stats, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "monthly_returns"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_yearly_returns(self, stats, savefig=False, **kwargs):
        '''
        Plots a barplot of returns by year.

        Parameters:
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
            
        plt.figure(figsize=(12,5))
        plot_yearly_returns(stats, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "yeayly_returns"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_txt_curve(self, stats, savefig=False, **kwargs):
        """
        Outputs the statistics for the equity curve.

        Parameters:
        stats = self.get_results()
        savefig = True or False
        """
        if stats is None:
            stats = self.get_results()
        plt.figure(figsize=(5,5))

        plot_txt_curve(stats, periods = self.periods ,**kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "txt_curve"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')



    def plot_txt_trade(self, stats, savefig=False, **kwargs):
        '''
        Outputs the statistics for the trades.

        Parameters:
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
        plt.figure(figsize=(5,5))

        plot_txt_trade(stats, freq = self.config['freq'], **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "txt_trade"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')


    def plot_txt_time(self, stats=None, savefig=False, **kwargs):
        '''
        Outputs the statistics for various time frames.

        Parameters:
        stats = self.get_results()
        savefig = True or False
        '''
        if stats is None:
            stats = self.get_results()
            
        plt.figure(figsize=(5,5))
        plot_txt_time(stats, **kwargs)

        if savefig:
            out_dir = self.config['out_dir']
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            now = datetime.utcnow()
            title = self.config['title'] + "_" + "txt_time"
            filename = out_dir + "\\" + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')

