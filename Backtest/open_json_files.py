import json
import pandas as pd
import numpy as np

def open_json_files(csv_dir, ticker):
    data = open(csv_dir).readlines()
    records = []
    times = []
    for line in data:
        j = json.loads(line)
        record = []
        record.append(j["volume"])
        record.append(j["last"])
        # record.append(pd.Timestamp(j["exchange_time"]))
        times.append(pd.Timestamp(j["exchange_time"]))
        records.append(record)
    df = pd.DataFrame.from_records(records, columns=["volume","last"], index=times)
    return df