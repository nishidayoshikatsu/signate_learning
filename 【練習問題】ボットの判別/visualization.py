# ライブラリの読み込み
import numpy as np
import pandas as pd
import pandas_profiling as pdp
from multiprocessing import Process, freeze_support

# dataの読み込み
df_train = pd.read_table("./data/train.tsv")
df_test = pd.read_table("./data/test.tsv")
print("complete reading data")
print(df_train)
print(df_train.shape)

if __name__ == '__main__':
    freeze_support()
    #Process(target=f).start()
    # EDA
    profile_train = pdp.ProfileReport(df_train)
    profile_test = pdp.ProfileReport(df_test)
    profile_train.to_file(outputfile="./EDA/train.html")
    profile_test.to_file(outputfile="./EDA/test.html")