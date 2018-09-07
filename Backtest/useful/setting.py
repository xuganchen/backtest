import sys
import os
import warnings
import json

with open('config.json') as configJsonFile:
    global back_config
    back_config = json.load(configJsonFile)


def set_env():
    backtest_dir = back_config['backtest_folder']
    if os.path.exists(backtest_dir):
        if backtest_dir not in sys.path:
            sys.path.insert(0, backtest_dir)
            print("The path '%s' is inserted into the sys.path" % backtest_dir)
        else:
            # print("The path '%s' is already in the sys.path" % backtest_dir)
            pass
    else:
        warnings.warn("The path '%s' does not exist" % backtest_dir)

    if not os.path.exists(back_config['output_folder']):
        os.makedirs(back_config['output_folder'])


    return back_config
