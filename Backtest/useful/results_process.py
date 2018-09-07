from setting import set_env
config = set_env()

import os
import sys
import pickle as pkl
import pandas as pd
import shutil
import glob

strategys = ['ADXStrategy', 'ADX_BOLLStrategy', 'ADX_EMVStrategy', 'ADX_RSIStrategy', 
             'AOStrategy', 'BNHStrategy', 'BOLLStrategy', 'BOLL_EMVStrategy', 'BOLL_RSIStrategy',
             'CCIStrategy', 'CMFStrategy', 'CMOStrategy', 'EMAStrategy', 'EMVStrategy', 
             'EMV_RSIStrategy', 'KDJStrategy', 'MACDCrossStrategy', 'MACDStrategy', 
             'MACD_ADXStrategy', 'MACD_BOLLStrategy', 'MACD_EMVStrategy', 'MACD_RSIStrategy', 
             'MAStrategy', 'RSIStrategy', 'SMAStrategy']

columns = ['parameters', 'tot_return', 'annual_return','cagr','max_drawdown', 'max_drawdown_duration', 
           'sharpe', 'sortino', 'IR', 'trading_num', 'win_pct', 'avg_trd_pct', 
           'avg_win_pct', 'avg_loss_pct', 'avg_dit', 'avg_com_imp']

results_in = pd.DataFrame(columns=columns, index=strategys)
results_out = pd.DataFrame(columns=columns, index=strategys)
rolling_28D_in = {}
rolling_28D_out = {}

for strategy in strategys:
    for dirpath, dirnames, filenames in os.walk(os.path.join(config['output_folder'], strategy)):
        for filename in filenames:
            if filename[-3:] == 'pkl':
                print(filename)
                file_dir = os.path.join(dirpath, filename)
                with open(file_dir, 'rb') as fr:
                    results = pkl.load(fr)
                start_date = results['config']['start_date']
                if start_date == pd.Timestamp("2018-02-01T00:0:00", freq = "60" + "T"):
                    rolling_28D_in[strategy] = results['rolling_28D']
                    results_in.loc[strategy] = [
                        filename[8:-22],
                        results['tot_return'],
                        results['annual_return'],
                        results['cagr'],
                        results['max_drawdown'],
                        results['max_drawdown_duration'],
                        results['sharpe'],
                        results['sortino'],
                        results['IR'],
                        results['trade_info']['trading_num'],
                        results['trade_info']['win_pct'],
                        results['trade_info']['avg_trd_pct'],
                        results['trade_info']['avg_win_pct'],
                        results['trade_info']['avg_loss_pct'],
                        results['trade_info']['avg_dit'],
                        results['trade_info']['avg_com_imp']
                    ]
                elif start_date == pd.Timestamp("2018-06-01T00:0:00", freq = "60" + "T"):
                    rolling_28D_out[strategy] = results['rolling_28D']
                    results_out.loc[strategy] = [
                        filename[8:-22],
                        results['tot_return'],
                        results['annual_return'],
                        results['cagr'],
                        results['max_drawdown'],
                        results['max_drawdown_duration'],
                        results['sharpe'],
                        results['sortino'],
                        results['IR'],
                        results['trade_info']['trading_num'],
                        results['trade_info']['win_pct'],
                        results['trade_info']['avg_trd_pct'],
                        results['trade_info']['avg_win_pct'],
                        results['trade_info']['avg_loss_pct'],
                        results['trade_info']['avg_dit'],
                        results['trade_info']['avg_com_imp']
                    ]
                    




##


tot_return_dir = os.path.join(config['output_folder'], 'tot_return')
sharpe_dir = os.path.join(config['output_folder'], 'sharpe')

if not os.path.exists(tot_return_dir):
    os.makedirs(tot_return_dir)
if not os.path.exists(sharpe_dir):
    os.makedirs(sharpe_dir)

for strategy in strategys:
    for dirpath, dirnames, filenames in os.walk(os.path.join(config['output_folder'], strategy)):
        res_tot_return = {}
        res_sharpe = {}
        for filename in filenames:
            if filename[-3:] == 'pkl':
                file_dir = os.path.join(dirpath, filename)
                with open(file_dir, 'rb') as fr:
                    results = pkl.load(fr)
                    res_tot_return[filename] = results['tot_return']
                    res_sharpe[filename] = results['sharpe']
        max_return = pd.Series(res_tot_return).idxmax()
        max_sharpe = pd.Series(res_sharpe).idxmax()
        try:
            shutil.copy(os.path.join(dirpath, max_return), tot_return_dir)
        except:
            pass
        try:
            for data in glob.glob(os.path.join(dirpath, max_return[8:-22] + "*")): 
                shutil.move(data, tot_return_dir) 
        except:
            pass
        try:
            shutil.copy(os.path.join(dirpath, max_sharpe), sharpe_dir) 
        except:
            pass   
        try:
            for data in glob.glob(os.path.join(dirpath, max_sharpe[8:-22] + "*")): 
                shutil.move(data, sharpe_dir) 
        except:
            pass