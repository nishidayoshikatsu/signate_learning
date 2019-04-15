# ライブラリの読み込み
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

features = {
        "id":                       "int",
        "statuses_count":           "int",      # ツイート数###
        "default_profile":          "bit",
        "default_profile_image":    "bit",
        "friends_count":            "int",
        "followers_count":          "int",
        "favourites_count":         "int",###正規化
        "geo_enabled":              "bit",      # 地理情報の有無
        "listed_count":             "int",###正規化
        "account_age_hours":        "int",
        "diversity":                "float",    # 多様性
        "mean_mins_between_tweets": "float",# ツイート間隔###正規化
        "mean_tweet_length":        "float",
        "mean_retweets":            "float",###正規化
        "reply_rate":               "float",
        "bot":                      "bit"       # 最終的に求める特徴量
}

# dataの読み込み
df_train = pd.read_table("./data/train.tsv")
df_test = pd.read_table("./data/test.tsv")
print("complete reading data")

# いらない特徴量の削除
df_train = df_train.drop(["account_age_hours", "friends_count", "mean_tweet_length"], axis=1)
df_test = df_test.drop(["account_age_hours", "friends_count", "mean_tweet_length"], axis=1)

### 学習 ###

# X_train、Y_train、X_testを作成
X_train = df_train.drop(["bot"], axis=1)   # 学習データ
Y_train = df_train["bot"]                  # 教師データ
X_test  = df_test

# 学習
forest = RandomForestClassifier()
forest.fit(X_train, Y_train)
Y_pred = forest.predict(X_test)

print(cross_val_score(forest, X_train, Y_train, cv=50))        # そのまま分割
stratifiedkfold = StratifiedKFold(n_splits=50)                   # 層化分割
print('Cross-validation scores: \n{}'.format(cross_val_score(forest, X_train, Y_train, cv=stratifiedkfold)))

# 提出データの作成
submission = pd.DataFrame({
        "id": df_test["id"],
        "bot": Y_pred})
submission.to_csv('./result/randomforest2.csv', index=False, header=False)      # headerを消さんとスコア出んよ