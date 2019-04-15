# ライブラリの読み込み
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# dataの読み込み
df_train = pd.read_table("./data/train.tsv")
df_test = pd.read_table("./data/test.tsv")
print("complete reading data")

### 学習 ###

# X_train、Y_train、X_testを作成
X_train = df_train.drop(["bot"], axis=1)   # 学習データ
Y_train = df_train["bot"]                  # 教師データ
X_test  = df_test

# 学習
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)

# 提出データの作成
submission = pd.DataFrame({
        "id": df_test["id"],
        "bot": Y_pred})
submission.to_csv('./result/knn1.csv', index=False, header=False)      # headerを消さんとスコア出んよ