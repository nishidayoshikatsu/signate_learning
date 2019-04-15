# ライブラリの読み込み
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import pandas_profiling as pdp
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier

# dataの読み込み
df_train = pd.read_table("./data/train.tsv")
df_test = pd.read_table("./data/test.tsv")
print("complete reading data")
print(df_train)
print(df_train.shape)
#print(df_train["id"])
print(df_train.columns)
print(df_train.ix[[0], [0]])
#print(del_columns)
#df_train.drop(columns=del_columns, inplace=True)        # 使用しない特徴量の削除
#df_test.drop(columns=del_columns, inplace=True)         # 使用しない特徴量の削除
"""
# EDA
profile_train = pdp.ProfileReport(df_train)
profile_test = pdp.ProfileReport(df_test)
profile_train.to_file(outputfile="./EDA/train.html")
profile_test.to_file(outputfile="./EDA/test.html")
"""

### 学習 ###

# X_train、Y_train、X_testを作成
X_train = df_train.drop(["bot"], axis=1)   # 学習データ
Y_train = df_train["bot"]                       # 教師データ
X_test  = df_test

# 学習
forest = RandomForestClassifier()
forest.fit(X_train, Y_train)
Y_pred = forest.predict(X_test)

# 提出データの作成
submission = pd.DataFrame({
        "id": df_test["id"],
        "bot": Y_pred})
submission.to_csv('./result/randomforest1.csv', index=False, header=False)