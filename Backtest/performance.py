import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import groupby
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import linregress
import os


class Performance(object):
    def __init__(self, portfolio_handler, data_handler, periods = 252):
        self.portfolio_handler = portfolio_handler
        self.data_handler = data_handler
        self.equity = {}
        self.periods = periods

    def update(self, timestamp, portfolio_handler):
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
        positions = self.portfolio_handler.all_positions
        if len(positions) == 0:
            return None
        else:
            return pd.DataFrame(positions)

    def get_results(self):
        # Equity
        res_equity = pd.Series(self.equity).sort_index()

        # Returns
        res_returns = res_equity.pct_change().fillna(0.0)

        # Rolling Annualized Sharpe
        rolling = res_returns.rolling(window = self.periods)
        res_rolling_sharpe = np.sqrt(self.periods) * (
            rolling.mean() / rolling.std()
        )

        # Cummulative Returns
        res_cum_returns = np.exp(np.log(1 + res_returns).cumsum())

        # Drawdown, max drawdown, max drawdown duration
        res_drawdown, res_max_dd, res_mdd_dur = self._create_drawdown(res_cum_returns)

        # Sharpe Ratio
        res_sharpe = np.sqrt(self.periods) * np.mean(res_returns) / np.std(res_returns)

        results = {}
        results['returns'] = res_returns
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

        return results

    def _plot_equity(self, stats, ax = None, log_scale = False, **kwargs):
        def format_two_dec(x, pos):
            return '%.2f' % x

        equity = stats['cum_returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_two_dec)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        equity.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.axhline(1.0, linestyle='--', color='black', lw=1)
        ax.set_ylabel('Cumulative returns')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        if log_scale:
            ax.set_yscale('log')

        return ax


    def _plot_rolling_sharpe(self, stats, ax=None, **kwargs):
        def format_two_dec(x, pos):
            return '%.2f' % x

        sharpe = stats['rolling_sharpe']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_two_dec)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        sharpe.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.axvline(sharpe.index[252], linestyle="dashed", c="gray", lw=2)
        ax.set_ylabel('Rolling Annualised Sharpe')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        return ax


    def _plot_drawdown(self, stats, ax=None, **kwargs):
        def format_perc(x, pos):
            return '%.0f%%' % x

        drawdown = stats['drawdown']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        underwater = -100 * drawdown
        underwater.plot(ax=ax, lw=2, kind='area', color='red', alpha=0.3, **kwargs)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
        ax.set_title('Drawdown (%)', fontweight='bold')
        return ax


    def _plot_monthly_returns(self, stats, ax=None, **kwargs):
        returns = stats['returns']
        if ax is None:
            ax = plt.gca()

        monthly_ret = aggregate_returns(returns, 'monthly')
        monthly_ret = monthly_ret.unstack()
        monthly_ret = np.round(monthly_ret, 3)
        monthly_ret.rename(
            columns={1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                     5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                     9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'},
            inplace=True
        )

        sns.heatmap(
            monthly_ret.fillna(0) * 100.0,
            annot=True,
            fmt="0.1f",
            annot_kws={"size": 8},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=cm.RdYlGn,
            ax=ax, **kwargs)
        ax.set_title('Monthly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('')

        return ax


    def _plot_yearly_returns(self, stats, ax=None, **kwargs):
        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats['returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')

        yly_ret = aggregate_returns(returns, 'yearly') * 100.0
        yly_ret.plot(ax=ax, kind="bar")
        ax.set_title('Yearly Returns (%)', fontweight='bold')
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.xaxis.grid(False)

        return ax


    def _plot_txt_curve(self, stats, ax=None, **kwargs):
        """
        Outputs the statistics for the equity curve.
        """
        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats["returns"]
        cum_returns = stats['cum_returns']

        trd_yr = 0
        # if 'positions' not in stats:
        #     trd_yr = 0
        # else:
        #     positions = stats['positions']
        #     trd_yr = positions.shape[0] / (
        #         (returns.index[-1] - returns.index[0]).days / 365.0
        #     )

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        tot_ret = cum_returns[-1] - 1.0
        years = len(cum_returns) / float(self.periods)
        cagr = (cum_returns[-1] ** (1.0 / years)) - 1.0
        sharpe = stats['sharpe']
        sortino = np.sqrt(self.periods) * (np.mean(returns)) / np.std(returns[returns < 0])
        slope, intercept, r_value, p_value, std_err = linregress(range(cum_returns.shape[0]), cum_returns)
        rsq = r_value**2
        dd = stats['drawdown']
        dd_max = stats['max_drawdown']
        dd_dur = stats['max_drawdown_duration']

        ax.text(0.25, 8.9, 'Total Return', fontsize=8)
        ax.text(7.50, 8.9, '{:.0%}'.format(tot_ret), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 7.9, 'CAGR', fontsize=8)
        ax.text(7.50, 7.9, '{:.2%}'.format(cagr), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 6.9, 'Sharpe Ratio', fontsize=8)
        ax.text(7.50, 6.9, '{:.2f}'.format(sharpe), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 5.9, 'Sortino Ratio', fontsize=8)
        ax.text(7.50, 5.9, '{:.2f}'.format(sortino), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 4.9, 'Annual Volatility', fontsize=8)
        ax.text(7.50, 4.9, '{:.2%}'.format(returns.std() * np.sqrt(252)), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 3.9, 'R-Squared', fontsize=8)
        ax.text(7.50, 3.9, '{:.2f}'.format(rsq), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 2.9, 'Max Daily Drawdown', fontsize=8)
        ax.text(7.50, 2.9, '{:.2%}'.format(dd_max), color='red', fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 1.9, 'Max Drawdown Duration', fontsize=8)
        ax.text(7.50, 1.9, '{:.0f}'.format(dd_dur), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 0.9, 'Trades per Year', fontsize=8)
        ax.text(7.50, 0.9, '{:.1f}'.format(trd_yr), fontweight='bold', horizontalalignment='right', fontsize=8)
        ax.set_title('Curve', fontweight='bold')

        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax

    def _plot_txt_trade(self, stats, ax=None, **kwargs):
        def format_perc(x, pos):
            return '%.0f%%' % x

        if ax is None:
            ax = plt.gca()

        num_trades = 0
        win_pct = "N/A"
        win_pct_str = "N/A"
        avg_trd_pct = "N/A"
        avg_win_pct = "N/A"
        avg_loss_pct = "N/A"
        max_win_pct = "N/A"
        max_loss_pct = "N/A"
        # if 'positions' not in stats:
        #     num_trades = 0
        #     win_pct = "N/A"
        #     win_pct_str = "N/A"
        #     avg_trd_pct = "N/A"
        #     avg_win_pct = "N/A"
        #     avg_loss_pct = "N/A"
        #     max_win_pct = "N/A"
        #     max_loss_pct = "N/A"
        # else:
        #     pos = stats['positions']
        #     num_trades = pos.shape[0]
        #     win_pct = pos[pos["trade_pct"] > 0].shape[0] / float(num_trades)
        #     win_pct_str = '{:.0%}'.format(win_pct)
        #     avg_trd_pct = '{:.2%}'.format(np.mean(pos["trade_pct"]))
        #     avg_win_pct = '{:.2%}'.format(np.mean(pos[pos["trade_pct"] > 0]["trade_pct"]))
        #     avg_loss_pct = '{:.2%}'.format(np.mean(pos[pos["trade_pct"] <= 0]["trade_pct"]))
        #     max_win_pct = '{:.2%}'.format(np.max(pos["trade_pct"]))
        #     max_loss_pct = '{:.2%}'.format(np.min(pos["trade_pct"]))

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        # TODO: Position class needs entry date
        max_loss_dt = 'TBD'  # pos[pos["trade_pct"] == np.min(pos["trade_pct"])].entry_date.values[0]
        avg_dit = '0.0'  # = '{:.2f}'.format(np.mean(pos.time_in_pos))

        ax.text(0.5, 8.9, 'Trade Winning %', fontsize=8)
        ax.text(9.5, 8.9, win_pct_str, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 7.9, 'Average Trade %', fontsize=8)
        ax.text(9.5, 7.9, avg_trd_pct, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 6.9, 'Average Win %', fontsize=8)
        ax.text(9.5, 6.9, avg_win_pct, fontsize=8, fontweight='bold', color='green', horizontalalignment='right')

        ax.text(0.5, 5.9, 'Average Loss %', fontsize=8)
        ax.text(9.5, 5.9, avg_loss_pct, fontsize=8, fontweight='bold', color='red', horizontalalignment='right')

        ax.text(0.5, 4.9, 'Best Trade %', fontsize=8)
        ax.text(9.5, 4.9, max_win_pct, fontsize=8, fontweight='bold', color='green', horizontalalignment='right')

        ax.text(0.5, 3.9, 'Worst Trade %', fontsize=8)
        ax.text(9.5, 3.9, max_loss_pct, color='red', fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 2.9, 'Worst Trade Date', fontsize=8)
        ax.text(9.5, 2.9, max_loss_dt, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 1.9, 'Avg Days in Trade', fontsize=8)
        ax.text(9.5, 1.9, avg_dit, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.text(0.5, 0.9, 'Trades', fontsize=8)
        ax.text(9.5, 0.9, num_trades, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.set_title('Trade', fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax

    def _plot_txt_time(self, stats, ax=None, **kwargs):
        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats['returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        mly_ret = aggregate_returns(returns, 'monthly')
        yly_ret = aggregate_returns(returns, 'yearly')

        mly_pct = mly_ret[mly_ret >= 0].shape[0] / float(mly_ret.shape[0])
        mly_avg_win_pct = np.mean(mly_ret[mly_ret >= 0])
        mly_avg_loss_pct = np.mean(mly_ret[mly_ret < 0])
        mly_max_win_pct = np.max(mly_ret)
        mly_max_loss_pct = np.min(mly_ret)
        yly_pct = yly_ret[yly_ret >= 0].shape[0] / float(yly_ret.shape[0])
        yly_max_win_pct = np.max(yly_ret)
        yly_max_loss_pct = np.min(yly_ret)

        ax.text(0.5, 8.9, 'Winning Months %', fontsize=8)
        ax.text(9.5, 8.9, '{:.0%}'.format(mly_pct), fontsize=8, fontweight='bold',
                horizontalalignment='right')

        ax.text(0.5, 7.9, 'Average Winning Month %', fontsize=8)
        ax.text(9.5, 7.9, '{:.2%}'.format(mly_avg_win_pct), fontsize=8, fontweight='bold',
                color='red' if mly_avg_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 6.9, 'Average Losing Month %', fontsize=8)
        ax.text(9.5, 6.9, '{:.2%}'.format(mly_avg_loss_pct), fontsize=8, fontweight='bold',
                color='red' if mly_avg_loss_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 5.9, 'Best Month %', fontsize=8)
        ax.text(9.5, 5.9, '{:.2%}'.format(mly_max_win_pct), fontsize=8, fontweight='bold',
                color='red' if mly_max_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 4.9, 'Worst Month %', fontsize=8)
        ax.text(9.5, 4.9, '{:.2%}'.format(mly_max_loss_pct), fontsize=8, fontweight='bold',
                color='red' if mly_max_loss_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 3.9, 'Winning Years %', fontsize=8)
        ax.text(9.5, 3.9, '{:.0%}'.format(yly_pct), fontsize=8, fontweight='bold',
                horizontalalignment='right')

        ax.text(0.5, 2.9, 'Best Year %', fontsize=8)
        ax.text(9.5, 2.9, '{:.2%}'.format(yly_max_win_pct), fontsize=8,
                fontweight='bold', color='red' if yly_max_win_pct < 0 else 'green',
                horizontalalignment='right')

        ax.text(0.5, 1.9, 'Worst Year %', fontsize=8)
        ax.text(9.5, 1.9, '{:.2%}'.format(yly_max_loss_pct), fontsize=8,
                fontweight='bold', color='red' if yly_max_loss_pct < 0 else 'green',
                horizontalalignment='right')

        # ax.text(0.5, 0.9, 'Positive 12 Month Periods', fontsize=8)
        # ax.text(9.5, 0.9, num_trades, fontsize=8, fontweight='bold', horizontalalignment='right')

        ax.set_title('Time', fontweight='bold')
        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax

    def plot_results(self, config = None):
        if config is None:
            self.title = config['title']
        else:
            self.title = None
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

        self._plot_equity(stats, ax=ax_equity)
        self._plot_rolling_sharpe(stats, ax=ax_sharpe)
        self._plot_drawdown(stats, ax=ax_drawdown)
        self._plot_monthly_returns(stats, ax=ax_monthly_returns)
        self._plot_yearly_returns(stats, ax=ax_yearly_returns)
        self._plot_txt_curve(stats, ax=ax_txt_curve)
        self._plot_txt_trade(stats, ax=ax_txt_trade)
        self._plot_txt_time(stats, ax=ax_txt_time)

        plt.show(block=False)
        if config is not None and config['save_plot'] == True:
            now = datetime.utcnow()
            out_dir = config['out_dir']
            title = config['title']
            filename = out_dir + title + "_" + now.strftime("%Y-%m-%d_%H%M%S") + ".png"
            fig.savefig(filename, dpi=150, bbox_inches='tight')



def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by day, week, month, or year.
    """
    def cumulate_returns(x):
        return np.exp(np.log(1 + x).cumsum())[-1] - 1

    if convert_to == 'weekly':
        return returns.groupby(
            [lambda x: x.year,
             lambda x: x.month,
             lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == 'monthly':
        return returns.groupby(
            [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == 'yearly':
        return returns.groupby(
            [lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError('convert_to must be weekly, monthly or yearly')
