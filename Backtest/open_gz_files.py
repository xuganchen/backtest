import os
import gzip
import pandas as pd

def open_gz_files(csv_dir, ticker):
    if ticker not in os.listdir(csv_dir):
        print("---------------------------------")
        print("%s is not in data list." % ticker)
        print("---------------------------------")
        return pd.DataFrame(columns=["volume", "last"])
    else:
        csv_dir = os.path.join(csv_dir, ticker)
        data_df = pd.DataFrame(columns=["volume", "last"])

        for dirpath, dirnames, filenames in os.walk(csv_dir):
            for filename in filenames:
                file_dir = os.path.join(dirpath, filename)
                g_file = gzip.open(file_dir)
                line = g_file.readline()
                records = []
                for line in g_file:
                    lineSegments = bytes.decode(line)[:-1].split(",")
                    record = [
                        float(lineSegments[5]),
                        float(lineSegments[4]),
                        pd.Timestamp.fromtimestamp(float(lineSegments[3]) / 1000)
                    ]
                    records.append(record)
                g_file.close()
                df = pd.DataFrame.from_records(records, columns=["volume", "last", "timestamp"], index="timestamp")
                data_df = pd.concat([data_df, df])
        data_df = data_df.sort_index()

        print("---------------------------------")
        print("Data for %s is prepared." %(ticker))
        print("---------------------------------")
        return data_df