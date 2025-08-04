from sklearn.model_selection import GridSearchCV
#  from thundersvm import SVC
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.base import clone
import pandas as pd
import joblib
from collections import Counter

'''
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
'''

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

def model_predict(model, test_path):
    test_df = pd.read_csv(test_path, sep="\t")

    X_test = test_df.drop(columns=["id", "label"])
    y_test = test_df["label"]

    pred = model.predict(X_test)

    print(classification_report(y_test, pred))

def train_and_evaluate(final_model, train_path, test_path):

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    X_train = train_df.drop(columns=["id", "label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["id", "label"])
    y_test = test_df["label"]

    # Binary labels
    y_early_train = y_train.apply(lambda x: 1 if x == "early" else 0)
    y_late_train = y_train.apply(lambda x: 1 if x == "late" else 0)

    svm_early = clone(final_model)
    #svm_late = clone(final_model)

    svm_early.fit(X_train, y_train)
    #svm_late.fit(X_train, y_late_train)

    # Predict
    pred_early = svm_early.predict(X_test)
    #pred_late = svm_late.predict(X_test)

    # Combine predictions one vs one
    #final_preds = []
    #for pe, pl in zip(pred_early, pred_late):
    #    if pe == 1 and pl == 0:
    #        final_preds.append("early")
    #    elif pe == 0 and pl == 1:
    #        final_preds.append("late")
    #    elif pe == 0 and pl == 0:
    #        final_preds.append("middle")
    #    else:
    #        print("error: gene classified as early and late")

    # Evaluate
    #
    # Modell speichern
    import os
    os.makedirs("../models", exist_ok=True)
    joblib.dump(svm_early, "../models/final_svm_model.pkl")


    print(classification_report(y_test, pred_early))

if __name__ == "__main__":
    # Load datasets
    train_path = "../data/new_test_data/split_dataset_us15/train_set.tsv"
    test_path = "../data/new_test_data/split_dataset_us15/test_set.tsv"

    #final_model = tuneSVM(train_df, test_df)
    final_model = SVC(kernel="poly", C=0.1, gamma=0.01, degree=2, class_weight="balanced")

    # Evaluieren (optional)
    train_and_evaluate(final_model, train_path, test_path)
