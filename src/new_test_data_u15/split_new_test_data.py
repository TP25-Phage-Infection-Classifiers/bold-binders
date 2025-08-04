import os
import pandas as pd
import matplotlib.pyplot as plt

def find_single_tsv(directory):
    """Gibt den Pfad zur ersten .tsv-Datei im angegebenen Verzeichnis zurück."""
    for file in os.listdir(directory):
        if file.endswith(".tsv"):
            return os.path.join(directory, file)
    raise FileNotFoundError(f"Keine TSV-Datei im Verzeichnis {directory} gefunden.")

def load_dataframe(filepath):
    """Lädt eine TSV-Datei in ein pandas DataFrame."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"Geladen: {filepath}, Form: {df.shape}")
    return df

def exclude_overlap(train_df, test_df, key):
    """Entfernt überlappende Einträge aus dem Trainingsdatensatz basierend auf einem Schlüssel."""
    train_df = train_df.copy()
    overlap_ids = set(train_df[key]) & set(test_df[key])
    if overlap_ids:
        print(f"{len(overlap_ids)} doppelte '{key}' wurden aus dem Trainingsset entfernt.")
        train_df = train_df[~train_df[key].isin(overlap_ids)]
    else:
        print("Keine Überlappungen zwischen Trainings- und Testdaten gefunden.")
    return train_df


def save_to_tsv(train_df, test_df, output_dir):
    """Speichert Trainings- und Test-DataFrames im angegebenen Verzeichnis als TSV-Dateien."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_set.tsv")
    test_path = os.path.join(output_dir, "test_set.tsv")
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)
    print(f"Trainingsdaten gespeichert unter: {train_path}")
    print(f"Testdaten gespeichert unter: {test_path}")

def plot_label_counts(train_df, test_df, label_col):
    """Erstellt ein Balkendiagramm zur Visualisierung der Label-Anzahl im Trainings- und Testset."""
    train_counts = train_df[label_col].value_counts()
    test_counts = test_df[label_col].value_counts()
    labels = sorted(set(train_counts.index).union(test_counts.index))
    bar_width = 0.35
    x = range(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(x, [train_counts.get(l, 0) for l in labels], width=bar_width, label="Train", color="lightgreen")
    plt.bar([i + bar_width for i in x], [test_counts.get(l, 0) for l in labels], width=bar_width, label="Test", color="salmon")
    plt.xticks([i + bar_width / 2 for i in x], labels)
    plt.title("Verteilung der Labels (Train vs Test)")
    plt.xlabel("Label")
    plt.ylabel("Anzahl")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_label_ratios(train_df, test_df, label_col):
    """Erstellt Kreisdiagramme zur Visualisierung der relativen Label-Häufigkeiten im Train- und Testset."""
    labels = sorted(set(train_df[label_col]) | set(test_df[label_col]))
    pie_colors = ['lightgray', 'lightblue', 'lightcoral', 'gold', 'violet', 'lightpink']

    def get_ratios(df):
        return df[label_col].value_counts(normalize=True)

    train_ratios = get_ratios(train_df)
    test_ratios = get_ratios(test_df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    datasets = [train_ratios, test_ratios]
    titles = ["Trainingsset - Labelverteilung", "Testset - Labelverteilung"]

    for ax, data, title in zip(axes, datasets, titles):
        ax.pie(
            [data.get(l, 0) for l in labels],
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors[:len(labels)]
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

def run_us15_split(
        train_dir="../data/feature_matrix",
        test_dir="../data/new_test_data/feature_matrix_test",
        output_dir="../data/new_test_data/split_dataset_us15",
        key_column="id",
        label_column="label"
):
    """Führt den US15-Datensatz-Split-Prozess inklusive Visualisierung durch."""

    # Schritt 1: Daten laden
    train_path = find_single_tsv(train_dir)
    test_path = find_single_tsv(test_dir)
    train_df = load_dataframe(train_path)
    test_df = load_dataframe(test_path)

    # Schritt 2: Überlappungen entfernen
    train_df = exclude_overlap(train_df, test_df, key=key_column)

    # Schritt 3: Ergebnis speichern
    save_to_tsv(train_df, test_df, output_dir)

    # Schritt 4: Verteilungen visualisieren
    plot_label_counts(train_df, test_df, label_col=label_column)
    plot_label_ratios(train_df, test_df, label_col=label_column)
