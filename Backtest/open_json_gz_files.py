import os
import gzip
import pandas as pd

def open_json_gz_files(gz_dir, ticker, start_date, end_date):
    # gz_dir: "...\Binance"
    # ticker: the name
    if ticker not in os.listdir(gz_dir):
        print("---------------------------------")
        print("%s is not in data list." % ticker)
        print("---------------------------------")
        return pd.DataFrame(columns=["volume", "last"])
    else:
        gz_dir = os.path.join(gz_dir, ticker)
        data_df = pd.DataFrame(columns=["volume", "last"])

        for dirpath, dirnames, filenames in os.walk(gz_dir):
            for filename in filenames:
                filedate = pd.Timestamp(filename[-18:-8])
                if filedate >= start_date and filedate <= end_date:
                    file_dir = os.path.join(dirpath, filename)
                    file_df = pd.read_json(file_dir, orient='records', lines=True, compression='gzip')
                    file_df = file_df.rename({"exchange_time": "timestamp"}, axis=1)
                    df = file_df[['volume', 'last', 'timestamp']]
                    df = df.set_index('timestamp')
                    data_df = pd.concat([data_df, df])
        data_df = data_df.sort_index()

        print("---------------------------------")
        print("Data for %s is prepared." %(ticker))
        print("---------------------------------")
        return data_df
