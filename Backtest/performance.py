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
    def __init__(self, config, portfolio_handler, data_handler, periods = 365):
        self.config = config
        self.portfolio_handler = portfolio_handler
        self.data_handler = data_handler
        self.equity = {}
        self.periods = periods

    def update(self, timestamp):
        self.equity[timestamp] = self.portfolio_handler.equity

    def _create_drawdown(self, cum_returns):
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
        # Equity
        res_equity = pd.Series(self.equity).sort_index()

        # Returns
        res_returns = res_equity.pct_change().fillna(0.0)
        res_daily_returns = np.exp(np.log(1 + res_returns).resample('D').sum().dropna())

        # Rolling Annualized Sharpe
        rolling = res_daily_returns.rolling(window = self.periods)
        res_rolling_sharpe = np.sqrt(self.periods) * (
            rolling.mean() / rolling.std()
        )

        # Cummulative Returns
        res_cum_returns = np.exp(np.log(1 + res_returns).cumsum())

        # Drawdown, max drawdown, max drawdown duration
        res_drawdown, res_max_dd, res_mdd_dur = self._create_drawdown(res_cum_returns)

        # Sharpe Ratio
        if np.std(res_daily_returns) == 0:
            res_sharpe = np.nan
        else:
            res_sharpe = np.sqrt(self.periods) * np.mean(res_daily_returns) / np.std(res_daily_returns)

        results = {}
        results['returns'] = res_returns
        results['daily_returns'] = res_daily_returns
        results['equity'] = res_equity
        results['rolling_sharpe'] = res_rolling_sharpe
        results['cum_returns'] = res_cum_returns
        results['drawdown'] = res_drawdown
        results['max_drawdown'] = res_max_dd
        results['max_drawdown_duration'] = res_mdd_dur
        results['sharpe'] = res_sharpe

        positions = self._get_positions()
        if positions is not None:
            results['positions'] = positions
            num_trades = positions.shape[0]
            win_pct = positions[positions["return"] > 0].shape[0] / float(num_trades)
            win_pct_str = '{:.0%}'.format(win_pct)
            avg_trd_pct = '{:.2%}'.format(np.mean(positions["return"]))
            avg_win_pct = '{:.2%}'.format(np.mean(positions[positions["return"] > 0]["return"]))
            avg_loss_pct = '{:.2%}'.format(np.mean(positions[positions["return"] <= 0]["return"]))
            max_win_pct = '{:.2%}'.format(np.max(positions["return"]))
            max_loss_pct = '{:.2%}'.format(np.min(positions["return"]))
            max_loss = positions[positions["return"] == np.min(positions["return"])]
            max_loss_dt = np.mean(max_loss["timedelta"])
            avg_dit = np.mean(positions["timedelta"])
            results['trade_info'] = {
                "trading_num": num_trades,      # 'Trades'
                "win_pct": win_pct_str,         # 'Trade Winning %'
                "avg_trd_pct": avg_trd_pct,     # 'Average Trade %'
                "avg_win_pct": avg_win_pct,     # 'Average Win %'
                "avg_loss_pct": avg_loss_pct,   # 'Average Loss %'
                "max_win_pct": max_win_pct,     # 'Best Trade %'
                "max_loss_pct": max_loss_pct,   # 'Worst Trade %'
                "max_loss_dt": max_loss_dt,     # 'Worst Trade Date'
                "avg_dit": avg_dit              # 'Avg Days in Trade'
            }
        else:
            results['trade_info'] = {
                "trading_num": 0,               # 'Trades'
                "win_pct": 'N/A',               # 'Trade Winning %'
                "avg_trd_pct": 'N/A',           # 'Average Trade %'
                "avg_win_pct": 'N/A',           # 'Average Win %'
                "avg_loss_pct": 'N/A',          # 'Average Loss %'
                "max_win_pct": 'N/A',           # 'Best Trade %'
                "max_loss_pct": 'N/A',          # 'Worst Trade %'
                "max_loss_dt": 0,               # 'Worst Trade Date'
                "avg_dit": 0                    # 'Avg Days in Trade'
            }

        return results


    def plot_results(self):
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
        vertical_sections = 6
        fig = plt.figure(figsize=(10, vertical_sections * 3.5))
        fig.suptitle(self.title, y=0.94, weight='bold')
        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.5)

        stats = self.get_results()
        ax_equity = plt.subplot(gs[:2, :])
        ax_sharpe = plt.subplot(gs[2, :])
        ax_drawdown = plt.subplot(gs[3, :])
        ax_monthly_returns = plt.subplot(gs[4, :2])
        ax_yearly_returns = plt.subplot(gs[4, 2])
        ax_txt_curve = plt.subplot(gs[5, 0])
        ax_txt_trade = plt.subplot(gs[5, 1])
        ax_txt_time = plt.subplot(gs[5, 2])

        plot_equity(stats, ax=ax_equity)
        plot_rolling_sharpe(stats, ax=ax_sharpe)
        plot_drawdown(stats, ax=ax_drawdown)
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



