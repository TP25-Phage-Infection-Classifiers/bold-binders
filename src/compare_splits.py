import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np

def load_data(train_path, test_path, label_encoder):
    """
    Lädt Trainings- und Testdaten (TSV) ein, entfernt unnötige Spalten
    und kodiert Zielklassen mit dem LabelEncoder.
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
    Trainiert Random Forest und gibt Vorhersagen sowie Wahrscheinlichkeiten zurück.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return y_pred, y_proba


def evaluate(y_test, y_pred, y_proba, report_dict, strategy_name, label_encoder):
    """
    Berechnet Accuracy, AUC, gewichtete Metriken sowie Precision, Recall und F1 für jede Klasse.
    """
    weighted = report_dict["weighted avg"]
    support = sum([v["support"] for k, v in report_dict.items() if k in label_encoder.classes_])

    try:
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            y_test_bin = label_binarize(y_test, classes=unique_classes)
            auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
    except:
        auc = float("nan")

    result = {
        "Strategy": strategy_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(auc, 4) if not np.isnan(auc) else "N/A",
        "Precision": round(weighted["precision"], 4),
        "Recall": round(weighted["recall"], 4),
        "F1-score": round(weighted["f1-score"], 4),
        "Support": support
    }

    # Klassenspezifische Scores hinzufügen
    for label in label_encoder.classes_:
        if label in report_dict:
            result[f"F1_{label}"] = round(report_dict[label]["f1-score"], 4)
            result[f"Precision_{label}"] = round(report_dict[label]["precision"], 4)
            result[f"Recall_{label}"] = round(report_dict[label]["recall"], 4)
        else:
            result[f"F1_{label}"] = result[f"Precision_{label}"] = result[f"Recall_{label}"] = "N/A"

    return result

def compute_difference_row(df, label="Difference"):
    """
    Berechnet Differenz zwischen zwei Strategien
    """
    row1, row2 = df.iloc[0], df.iloc[1]
    diff_row = {"Strategy": label}
    for col in df.columns:
        if col != "Strategy":
            try:
                diff_row[col] = round(row1[col] - row2[col], 4)
            except:
                diff_row[col] = "N/A"
    return diff_row


def plot_heatmap(df):
    """
    Erstellt eine Heatmap zur Visualisierung aller numerischen Klassifikationsmetriken,
    inklusive klassenspezifischer Precision, Recall und F1-Scores.
    """
    # Nur numerische Spalten extrahieren, Strategie als Index
    df_plot = df.set_index("Strategy").select_dtypes(include=[float, int])

    # Dynamische Breite je nach Anzahl Metriken
    num_cols = df_plot.shape[1]
    fig_width = max(12, num_cols * 0.7)

    plt.figure(figsize=(fig_width, 4))
    sns.heatmap(df_plot, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Strategievergleich inkl. klassenspezifischer Metriken")
    plt.tight_layout()
    plt.show()

def run_comparison(train1_path, test1_path, train2_path, test2_path, label_encoder):
    """
    Führt vollständigen Vergleich zweier Strategien durch:
    - Laden der Daten
    - Training & Vorhersage
    - Klassifikationsbericht
    - Evaluation
    - Differenzberechnung
    - Rückgabe des vollständigen Ergebnis-DataFrames
    """
    # Strategie 1
    X_train1, y_train1, X_test1, y_test1 = load_data(train1_path, test1_path, label_encoder)
    y_pred1, y_proba1 = train_and_predict(X_train1, y_train1, X_test1)
    report1 = classification_report(y_test1, y_pred1, target_names=label_encoder.classes_, output_dict=True)
    result1 = evaluate(y_test1, y_pred1, y_proba1, report1, "Strategy 1 (US14)", label_encoder)

    # Strategie 2
    X_train2, y_train2, X_test2, y_test2 = load_data(train2_path, test2_path, label_encoder)
    y_pred2, y_proba2 = train_and_predict(X_train2, y_train2, X_test2)
    report2 = classification_report(y_test2, y_pred2, target_names=label_encoder.classes_, output_dict=True)
    result2 = evaluate(y_test2, y_pred2, y_proba2, report2, "Strategy 2 (US15)", label_encoder)

    # Zusammenführen und Differenz berechnen
    results_df = pd.DataFrame([result1, result2])
    diff_row = compute_difference_row(results_df)
    results_df = pd.concat([results_df, pd.DataFrame([diff_row])], ignore_index=True)

    return results_df




