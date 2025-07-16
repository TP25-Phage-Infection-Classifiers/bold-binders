from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from collections import Counter

def supportVectorMachine(train_df, test_df):

    print(train_df.shape)
    print(test_df.shape)
    print(train_df.head())

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

    #Combine predictions one vs one
    final_preds = []
    for pe, pl in zip(pred_early, pred_late):
        if pe == 1 and pl == 0:
            final_preds.append("early")
        elif pe == 0 and pl == 1:
            final_preds.append("late")
        elif pe == 0 and pl == 0:
            final_preds.append("middle")
        else:
            print("error: gene classified as early and late")

    #Evaluate
    print(classification_report(y_test, final_preds))

    #multi class prediction
    svm_multi = SVC(kernel="rbf")
    svm_multi.fit(X_train, y_train)
    preds = svm_multi.predict(X_test)
    print(classification_report(y_test, preds))


def tuneSVM(train_df, test_df):
    print(train_df.shape)
    print(test_df.shape)
    print(train_df.head())

    # Prepare features and labels
    X_train = train_df.drop(columns=["id", "label"]).astype("float32")
    X_test = test_df.drop(columns=["id", "label"]).astype("float32")
    y_train = train_df["label"]
    y_test = test_df["label"]

    # Define parameter grid with varying kernels
    param_grid = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10, 100],
            "class_weight": [None, "balanced"],
            "decision_function_shape": ["ovr"]
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10, 100],
            "gamma": [0.001, 0.01, 0.1, 1],
            "class_weight": [None, "balanced"],
            "decision_function_shape": ["ovr"]
        },
        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10],
            "gamma": [0.001, 0.01],
            "degree": [2, 3],
            "class_weight": [None, "balanced"],
            "decision_function_shape": ["ovr"]
        },
        {
            "kernel": ["sigmoid"],
            "C": [0.1, 1, 10],
            "gamma": [0.001, 0.01, 0.1],
            "class_weight": [None, "balanced"],
            "decision_function_shape": ["ovr"]
        }
    ]

    svc = SVC()
    grid_search = GridSearchCV(
        svc,
        param_grid,
        cv=5,
        scoring="f1_macro",
        verbose=3,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Beste Parameter:", grid_search.best_params_)
    print("Bester Score:", grid_search.best_score_)

    # Beste Modell nehmen und auf Testdaten anwenden
    best_svm = grid_search.best_estimator_
    preds = best_svm.predict(X_test)

    print("Vorhersagen:", Counter(preds))
    print(classification_report(y_test, preds))

#Load pre-split datasets
train_df = pd.read_csv("../data/feature_matrix/feature_matrix.tsv", sep="\t")
test_df = pd.read_csv("../data/new_test_data/feature_matrix_test/feature_matrix.tsv", sep="\t")

# Load datasets
train_df_us15 = pd.read_csv("../data/new_test_data/split_dataset_us15/train_set.tsv", sep="\t")
test_df_us15 = pd.read_csv("../data/new_test_data/split_dataset_us15/test_set.tsv", sep="\t")

#supportVectorMachine(train_df_us15, test_df_us15)
#tuneSVM(train_df_us15, test_df_us15)