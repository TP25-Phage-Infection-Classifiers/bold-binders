import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np

def load_data(train_path, test_path, label_encoder):
    """
    Lädt Trainings- und Testdaten aus TSV-Dateien, entfernt nicht benötigte Spalten
    und wendet Label-Encoding auf die Zielvariablen an.
    """
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    X_train = train_df.drop(columns=["id", "label"])
    y_train = label_encoder.fit_transform(train_df["label"])

    X_test = test_df.drop(columns=["id", "label"])
    y_test = label_encoder.transform(test_df["label"])

    return X_train, y_train, X_test, y_test


def train_and_predict(X_train, y_train, X_test):
    """
    Trainiert ein Random-Forest-Modell mit den Trainingsdaten
    und erzeugt Vorhersagen für die Testdaten.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return y_pred, y_proba


def evaluate(y_test, y_pred, y_proba, report_dict, strategy_name):
    """
    Berechnet verschiedene Klassifikationsmetriken auf Basis der Vorhersagen.
    Unterstützt sowohl binäre als auch Multiklassen-Klassifikation.
    """
    weighted = report_dict["weighted avg"]
    support = sum([v["support"] for k, v in report_dict.items() if k.isdigit()])

    # AUC-Berechnung (robust für Binary und Multiclass)
    auc = float("nan")  # Default-Wert

    try:
        unique_classes = np.unique(y_test)
        n_classes = len(unique_classes)

        if n_classes == 2:
            # Binary classification
            if y_proba.ndim == 2:
                # [n_samples, n_classes] -> Klasse 1 확률만
                y_proba_bin = y_proba[:, 1]
            else:
                y_proba_bin = y_proba
            auc = roc_auc_score(y_test, y_proba_bin)

        elif n_classes > 2:
            # Multiclass classification
            if y_proba.ndim != 2 or y_proba.shape[1] != n_classes:
                raise ValueError(f"Multiclass AUC: Erwartete shape (n_samples, {n_classes}), bekam {y_proba.shape}")
            y_test_bin = label_binarize(y_test, classes=unique_classes)
            auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")

    except Exception as e:
        print(f"[AUC-Berechnung fehlgeschlagen] {e}")
        auc = float("nan")

    return {
        "Strategy": strategy_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
        "Precision": round(weighted["precision"], 4),
        "Recall": round(weighted["recall"], 4),
        "F1-score": round(weighted["f1-score"], 4),
        "Support": support
    }

def evaluate_strategy(train_path, test_path, label_encoder, strategy_name):
    X_train, y_train, X_test, y_test = load_data(train_path, test_path, label_encoder)
    y_pred, y_proba = train_and_predict(X_train, y_train, X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return evaluate(y_test, y_pred, y_proba, report, strategy_name)


def compute_difference_row(df, label="Difference"):
    metrics = ["Accuracy", "AUC", "Precision", "Recall", "F1-score", "Support"]
    row1 = df.iloc[0]
    row2 = df.iloc[1]
    diff_row = {"Strategy": label}
    for metric in metrics:
        try:
            diff = row1[metric] - row2[metric]
            diff_row[metric] = round(diff, 4)
        except:
            diff_row[metric] = "N/A"
    return diff_row

def plot_heatmap(df):
    """
    Erstellt eine Heatmap zur Visualisierung der Klassifikationsmetriken
    für verschiedene Strategien.
    """
    df_plot = df.set_index("Strategy")[["Accuracy", "AUC", "Precision", "Recall", "F1-score"]]
    plt.figure(figsize=(8, 3))
    sns.heatmap(df_plot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, linewidths=0.5)
    plt.title("Leistungsvergleich der Split-Strategien")
    plt.tight_layout()
    plt.show()



