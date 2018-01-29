import os
import pandas as pd


def load_or_download_df(fpath, url=None):
    if os.path.exists(fpath):
        print("-- found locally")
        df = pd.read_csv(fpath, index_col=0)
    else:
        print("-- trying to download from url")
        try:
            df = pd.read_csv(url)
        except:
            exit("-- Unable to download from url")

        with open(fpath, 'w') as f:
            print("-- writing to local file file")
            df.to_csv(f)
    return df