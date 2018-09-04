
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns



def plot_equity(stats, config, ax=None, log_scale=False, mid_time=None, **kwargs):
    '''
    Plots cumulative rolling returns
    '''
    def format_two_dec(x, pos):
        return '%.2f' % x
    def format_perc(x, pos):
        return '%.0f%%' % x

    equity = stats['cum_returns'] * 100
    BNH_equity = stats['BNH_cum_returns'] * 100

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_perc)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.xaxis.set_tick_params(reset=True)
    ax.yaxis.grid(linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.grid(linestyle=':')

    equity.plot(lw=2, color='blue', alpha=0.6, x_compat=False,
                label=config['title'], ax=ax, **kwargs)
    BNH_equity.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                label='Buy and Hold Strategy', ax=ax, **kwargs)

    end_time = equity.index[-1]
    ax.axhline(equity[end_time], linestyle='--', color='blue', lw=1)
    ax.axhline(BNH_equity[end_time], linestyle='--', color='green', lw=1)

    if mid_time is not None:
        ax.axvline(mid_time, linestyle='--', color='red', lw=1)
        ax.axhline(equity[mid_time], linestyle='--', color='blue', lw=1)
        ax.axhline(BNH_equity[mid_time], linestyle='--', color='green', lw=1)

    ax.axhline(1.0, linestyle='--', color='black', lw=1)
    ax.set_ylabel('')
    ax.legend(loc='best')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
    ax.set_title('Cumulative Returns (%)', fontweight='bold')

    if log_scale:
        ax.set_yscale('log')

    return ax


def plot_rolling_sharpe(stats, ax=None, mid_time=None, **kwargs):
    '''
    Plots the curve of rolling Sharpe ratio.
    '''
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

    if mid_time is not None:
        if mid_time > sharpe.index[0]:
            pass
        else:
            mid_time = sharpe.index[0]
        ax.axvline(mid_time, linestyle='--', color='green', lw=1)
        ax.axhline(sharpe[mid_time], linestyle='--', color='green', lw=1)

    ax.set_ylabel('')
    ax.legend(loc='best')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
    ax.set_title('Rolling Annualised Sharpe', fontweight='bold')

    return ax


def plot_drawdown(stats, ax=None, mid_time=None, **kwargs):
    '''
    Plots the underwater curve
    '''
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

    if mid_time is not None:
        ax.axvline(mid_time, linestyle='--', color='red', lw=1)
        ax.axhline(underwater[mid_time], linestyle='--', color='red', lw=1)

    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
    ax.set_title('Drawdown (%)', fontweight='bold')
    return ax


def plot_weekly_returns(stats, ax=None, **kwargs):
    '''
    Plots a bar of the weekly returns.
    '''
    def format_perc(x, pos):
        return '%.0f%%' % x

    rolling_return_week = stats['rolling_return_week']

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_perc)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.yaxis.grid(linestyle=':')

    wly_ret = rolling_return_week * 100.0
    wly_ret.plot(ax=ax, kind="bar")
    # x = range(wly_ret.shape[0])
    # y = wly_ret.values
    # for a, b in zip(x, y):
    #     if b >= 0:
    #         ax.text(a, b+0.01, "%.3f" % b, ha='center', va='bottom', fontsize=7)
    #     else:
    #         ax.text(a, b-0.1, "%.3f" % b, ha='center', va='bottom', fontsize=7)
    ax.set_title('Weekly Returns (%)', fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.xaxis.grid(False)

    return ax

def plot_monthly_returns(stats, ax=None, **kwargs):
    '''
    Plots a heatmap of the monthly returns.
    '''

    rolling_return_month = stats['rolling_return_month']

    if ax is None:
        ax = plt.gca()

    monthly_ret = rolling_return_month.unstack()
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
        fmt="0.001f",
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


def plot_yearly_returns(stats, ax=None, **kwargs):
    '''
    Plots a barplot of returns by year.
    '''
    def format_perc(x, pos):
        return '%.0f%%' % x

    rolling_return_year = stats['rolling_return_year']

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_perc)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.yaxis.grid(linestyle=':')

    yly_ret = rolling_return_year * 100.0
    yly_ret.plot(ax=ax, kind="bar")
    # x = range(yly_ret.shape[0])
    # y = yly_ret.values
    # for a, b in zip(x, y):
    #     if b >= 0:
    #         ax.text(a, b+0.01, "%.3f" % b, ha='center', va='bottom', fontsize=7)
    #     else:
    #         ax.text(a, b-0.1, "%.3f" % b, ha='center', va='bottom', fontsize=7)

    ax.set_title('Yearly Returns (%)', fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.xaxis.grid(False)

    return ax


def plot_txt_curve(stats, ax=None, periods = 365, **kwargs):
    """
    Outputs the statistics for the equity curve.
    """

    def format_perc(x, pos):
        return '%.0f%%' % x

    returns = stats["returns"]
    daily_returns = stats['daily_returns']
    cum_returns = stats['cum_returns']
    daily_cum_returns = stats['daily_cum_returns']

    positions = stats['positions']
    if positions is not None:
        trd_yr = positions.shape[0] / (
                (returns.index[-1] - returns.index[0]).days / (periods)
        )
    else:
        trd_yr = 0

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_perc)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    tot_ret = stats['tot_return']
    IR = stats['IR']
    sharpe = stats['sharpe']
    sortino = stats['sortino']
    slope, intercept, r_value, p_value, std_err = linregress(range(daily_cum_returns.shape[0]), daily_cum_returns)
    rsq = r_value ** 2
    dd = stats['drawdown']
    dd_max = stats['max_drawdown']
    dd_dur = stats['max_drawdown_duration']

    ax.text(0.25, 8.9, 'Total Return', fontsize=8)
    ax.text(7.50, 8.9, '{:.3%}'.format(tot_ret), fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 7.9, 'Infor Ratio', fontsize=8)
    ax.text(7.50, 7.9, '{:.3%}'.format(IR), fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 6.9, 'Sharpe Ratio', fontsize=8)
    ax.text(7.50, 6.9, '{:.3f}'.format(sharpe), fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 5.9, 'Sortino Ratio', fontsize=8)
    ax.text(7.50, 5.9, '{:.3f}'.format(sortino), fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 4.9, 'Annual Volatility', fontsize=8)
    ax.text(7.50, 4.9, '{:.3%}'.format(daily_returns.std() * np.sqrt(365)), fontweight='bold', horizontalalignment='right',
            fontsize=8)

    ax.text(0.25, 3.9, 'R-Squared', fontsize=8)
    ax.text(7.50, 3.9, '{:.3f}'.format(rsq), fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 2.9, 'Max Daily Drawdown', fontsize=8)
    ax.text(7.50, 2.9, '{:.3%}'.format(dd_max), color='red', fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 1.9, 'Max Drawdown Duration', fontsize=8)
    ax.text(7.50, 1.9, '{:.0f}'.format(dd_dur), fontweight='bold', horizontalalignment='right', fontsize=8)

    ax.text(0.25, 0.9, 'Trades per Year', fontsize=8)
    ax.text(7.50, 0.9, '{:.3f}'.format(trd_yr), fontweight='bold', horizontalalignment='right', fontsize=8)
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


def plot_txt_trade(stats, freq = 1, ax=None, **kwargs):
    '''
    Outputs the statistics for the trades.
    '''
    def format_perc(x, pos):
        return '%.0f%%' % x

    if ax is None:
        ax = plt.gca()

    trade_info = stats['trade_info']
    num_trades = trade_info['trading_num']
    win_pct_str = '{:.3%}'.format(trade_info['win_pct'])
    avg_trd_pct = '{:.3%}'.format(trade_info['avg_trd_pct'])
    avg_win_pct = '{:.3%}'.format(trade_info['avg_win_pct'])
    avg_loss_pct = '{:.3%}'.format(trade_info['avg_loss_pct'])
    max_win_pct = '{:.3%}'.format(trade_info['max_win_pct'])
    max_loss_pct = '{:.3%}'.format(trade_info['max_loss_pct'])
    max_loss_dt = trade_info['max_loss_dt']
    avg_dit = trade_info['avg_dit']

    y_axis_formatter = FuncFormatter(format_perc)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

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


def plot_txt_time(stats, ax=None, **kwargs):
    '''
    Outputs the statistics for various time frames.
    '''
    def format_perc(x, pos):
        return '%.0f%%' % x

    returns = stats['returns']
    rolling_return_month = stats['rolling_return_month']
    rolling_return_year = stats['rolling_return_year']

    if ax is None:
        ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_perc)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    mly_ret = rolling_return_month
    yly_ret = rolling_return_year

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
    ax.text(9.5, 7.9, '{:.3%}'.format(mly_avg_win_pct), fontsize=8, fontweight='bold',
            color='red' if mly_avg_win_pct < 0 else 'green',
            horizontalalignment='right')

    ax.text(0.5, 6.9, 'Average Losing Month %', fontsize=8)
    ax.text(9.5, 6.9, '{:.3%}'.format(mly_avg_loss_pct), fontsize=8, fontweight='bold',
            color='red' if mly_avg_loss_pct < 0 else 'green',
            horizontalalignment='right')

    ax.text(0.5, 5.9, 'Best Month %', fontsize=8)
    ax.text(9.5, 5.9, '{:.3%}'.format(mly_max_win_pct), fontsize=8, fontweight='bold',
            color='red' if mly_max_win_pct < 0 else 'green',
            horizontalalignment='right')

    ax.text(0.5, 4.9, 'Worst Month %', fontsize=8)
    ax.text(9.5, 4.9, '{:.3%}'.format(mly_max_loss_pct), fontsize=8, fontweight='bold',
            color='red' if mly_max_loss_pct < 0 else 'green',
            horizontalalignment='right')

    ax.text(0.5, 3.9, 'Winning Years %', fontsize=8)
    ax.text(9.5, 3.9, '{:.0%}'.format(yly_pct), fontsize=8, fontweight='bold',
            horizontalalignment='right')

    ax.text(0.5, 2.9, 'Best Year %', fontsize=8)
    ax.text(9.5, 2.9, '{:.3%}'.format(yly_max_win_pct), fontsize=8,
            fontweight='bold', color='red' if yly_max_win_pct < 0 else 'green',
            horizontalalignment='right')

    ax.text(0.5, 1.9, 'Worst Year %', fontsize=8)
    ax.text(9.5, 1.9, '{:.3%}'.format(yly_max_loss_pct), fontsize=8,
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

