import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from collections import Counter


def load_data(train_path, test_path):
    """
    Lädt Trainings- und Testdaten aus TSV-Dateien.
    """
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    return train_df, test_df


def preprocess_data(train_df, test_df):
    """
    Führt Vorverarbeitung durch: Trennt Features und Label, wendet Label-Encoding an.
    """
    X_train = train_df.drop(columns=["id", "label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["id", "label"])
    y_test = test_df["label"]

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    return X_train, y_train_enc, X_test, y_test_enc, label_encoder


def grid_search_decision_tree_with_smote(X_train, y_train_enc):
    """
    Führt eine Grid Search mit einem DecisionTreeClassifier durch,
    eingebettet in eine Pipeline mit SMOTE zur Behandlung von Klassenungleichgewichten.
    """
    # Ursprüngliche Klassenverteilung anzeigen
    class_counts = Counter(y_train_enc)
    print("Original class distribution:", class_counts)

    # SMOTE-Strategie definieren: late und middle leicht überabgetastet
    smote_strategy = {
        1: max(class_counts[1] + 50, class_counts[1]),  # late
        2: max(class_counts[2] + 50, class_counts[2])  # middle
    }

    print("Using SMOTE sampling strategy:", smote_strategy)

    # Pipeline mit SMOTE und Entscheidungsbaum
    pipeline = Pipeline([
        ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])

    # Hyperparameter-Raster für Grid Search
    param_grid = {
        "clf__max_depth": [5, 8, 10],
        "clf__min_samples_split": [5, 10],
        "clf__min_samples_leaf": [2, 4],
        "clf__criterion": ["gini", "entropy"],
        "clf__max_features": ["sqrt", "log2"]
    }

    # GridSearchCV zur Optimierung des Entscheidungsbaums
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train_enc)
    return grid_search.best_estimator_, grid_search.best_params_


def print_model_metrics(model, X, y_true, label_encoder):
    """
    Bewertet das Modell und gibt Metriken aus.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)
    return y_pred


def plot_confusion_matrices(y_train_true, y_train_pred, y_test_true, y_test_pred, label_encoder):
    """
    Zeigt die Confusion Matrix für Trainings- und Testdaten nebeneinander.
    """
    cm_train = confusion_matrix(y_train_true, y_train_pred)
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    labels = label_encoder.classes_

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Train Set")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Test Set")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.suptitle("Confusion Matrices (Train vs Test)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_top_features(model, X_train):
    """
    Visualisiert die wichtigsten Features nach Wichtigkeit.
    """
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    feature_names = X_train.columns

    top_indices = np.argsort(importances)[-20:][::-1]
    top_features = feature_names[top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features)
    plt.title("Top 20 Important Features (Decision Tree)")
    plt.xlabel("Feature Importance")
    plt.tight_layout()
    plt.show()


# === Hauptausführung ===
#if __name__ == "__main__":
    #train_path = "../data/split_dataset_us14/train_set.tsv"
    #test_path = "../data/split_dataset_us14/test_set.tsv"

    #train_df, test_df = load_data(train_path, test_path)
    #X_train, y_train_enc, X_test, y_test_enc, label_encoder = preprocess_data(train_df, test_df)

    #best_model, best_params = grid_search_decision_tree_with_smote(X_train, y_train_enc)
    #print("Beste Parameter:", best_params)

    #print("\nTestdatenbewertung:")
    #y_pred_test = print_model_metrics(best_model, X_test, y_test_enc, label_encoder)

    #print("\nTrainingsdatenbewertung (Pipeline mit SMOTE):")
    #y_pred_train = print_model_metrics(best_model, X_train, y_train_enc, label_encoder)

    #plot_confusion_matrices(y_train_enc, y_pred_train, y_test_enc, y_pred_test, label_encoder)

    #plot_top_features(best_model, X_train)
