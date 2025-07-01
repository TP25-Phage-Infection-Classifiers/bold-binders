'''
import pandas as pd
import numpy as np
from sklearn import svc

def svm():

    df = pd.read_csv("../data/feature_matrix/feature_matrix.tsv", sep="\t")

    feature_array = np.asarray(df)
    class_array = df.iloc[:, 1]

    #print(df.shape)

    clf = svc.SVC()

    clf.fit(feature_array, class_array)

svm()
'''
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report


#------------------------svm
#Load pre-split datasets
train_df = pd.read_csv("../data/split_data_bb/feature_training.tsv", sep="\t")
test_df = pd.read_csv("../data/split_data_bb/feature_test.tsv", sep="\t")

#Prepare features and labels
X_train = train_df.drop(columns=["id", "label"]).astype("float32")
X_test = test_df.drop(columns=["id", "label"]).astype("float32")
y_train = train_df["label"]
y_test = test_df["label"]

#Binary labels
y_early_train = y_train.apply(lambda x: 1 if x == "early" else 0)
y_late_train = y_train.apply(lambda x: 1 if x == "late" else 0)

#Train SVMs (non-linear kernel)
svm_early = SVC(kernel="rbf")
svm_late = SVC(kernel="rbf")

svm_early.fit(X_train, y_early_train)
svm_late.fit(X_train, y_late_train)

#Predict
pred_early = svm_early.predict(X_test)
pred_late = svm_late.predict(X_test)

#Combine predictions
final_preds = []
for pe, pl in zip(pred_early, pred_late):
    if pe == 1 and pl == 0:
        final_preds.append("early")
    elif pe == 0 and pl == 1:
        final_preds.append("late")
    else:
        final_preds.append("middle")

#Evaluate
print(classification_report(y_test, final_preds))