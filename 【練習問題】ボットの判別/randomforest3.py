# ライブラリの読み込み
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

features = {
        "id":                       "int",
        "statuses_count":           "int",      # ツイート数
        "default_profile":          "bit",
        "default_profile_image":    "bit",
        "friends_count":            "int",
        "followers_count":          "int",
        "favourites_count":         "int",
        "geo_enabled":              "bit",      # 地理情報の有無
        "listed_count":             "int",
        "account_age_hours":        "int",
        "diversity":                "float",    # 多様性
        "mean_mins_between_tweets": "float",    # ツイート間隔
        "mean_tweet_length":        "float",
        "mean_retweets":            "float",
        "reply_rate":               "float",
        "bot":                      "bit"       # 最終的に求める特徴量
}

# dataの読み込み
df_train = pd.read_table("./data/train.tsv")
df_test = pd.read_table("./data/test.tsv")
print("complete reading data")

def min_max(n):
    n_min = np.min(n)
    n_max = np.max(n)
    result = (n - n_min)/(n_max - n_min)
    return result

### 前処理 ###
features_list = list(features)
for feature in features_list[1:(len(features_list)-1)]:
    df_train[feature] = min_max(df_train[feature])

### 学習 ###

# X_train、Y_train、X_testを作成
X_train = df_train.drop(["bot"], axis=1)   # 学習データ
Y_train = df_train["bot"]                  # 教師データ
X_test  = df_test

"""
param_grid = {
        "max_depth": [2,3, None],
        "n_estimators":[50,100,200,500],
        "max_features": [1, 3, 10],
        "min_samples_split": [2, 3, 10],
        "min_samples_leaf": [1, 3, 10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
}

forest = GridSearchCV(
                estimator=RandomForestClassifier(random_state=0),
                param_grid = param_grid,   
                scoring="accuracy",  #metrics
                cv = 3,              #cross-validation
                n_jobs = 1          #number of core
                )
"""

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
submission.to_csv('./result/randomforest3.csv', index=False, header=False)      # headerを消さんとスコア出んよ