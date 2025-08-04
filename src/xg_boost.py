from sklearn.metrics import classification_report
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

train_df_us15 = pd.read_csv("../data/new_test_data/split_dataset_us15/train_set.tsv", sep="\t")
test_df_us15 = pd.read_csv("../data/new_test_data/split_dataset_us15/test_set.tsv", sep="\t")

print(train_df_us15.shape)
print(test_df_us15.shape)
print(train_df_us15.head())

X_train = train_df_us15.drop(columns=["id", "label"]).astype("float32")
X_test = test_df_us15.drop(columns=["id", "label"]).astype("float32")
y_train = train_df_us15["label"]
y_test = test_df_us15["label"]

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
print(clf.score(X_test, y_test))
