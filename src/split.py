import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def modified_find_tsv_file(directory):
    """Gibt den Pfad zur ersten .tsv-Datei im angegebenen Verzeichnis zurück."""
    tsv_files = [f for f in os.listdir(directory) if f.endswith(".tsv")]
    if not tsv_files:
        raise FileNotFoundError("Keine .tsv-Datei im Verzeichnis gefunden.")
    return os.path.join(directory, tsv_files[0])

def load_tsv(filepath):
    """Lädt eine TSV-Datei als pandas DataFrame."""
    df = pd.read_csv(filepath, sep="\t")
    print(f"Datei geladen: {filepath}")
    return df

def split_stratified(df, label_column='label', test_size=0.2, random_seed=42):
    """Teilt den DataFrame stratifiziert in Trainings- und Testdaten basierend auf dem Label."""
    return train_test_split(df, test_size=test_size, stratify=df[label_column], random_state=random_seed)

def save_splits_to_tsv(train_df, test_df, output_dir="../data/split_dataset_us14", train_file="train_set.tsv", test_file="test_set.tsv"):
    """Speichert Trainings- und Test-Daten als TSV-Dateien im angegebenen Verzeichnis."""
    os.makedirs(output_dir, exist_ok=True)  # Erstellt das Verzeichnis, falls es nicht existiert
    train_path = os.path.join(output_dir, train_file)
    test_path = os.path.join(output_dir, test_file)
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)
    print(f"\nTrainingsdaten gespeichert unter: {train_path}")
    print(f"Testdaten gespeichert unter: {test_path}")

def print_label_distribution(df, train_df, test_df, label_column='label'):
    """Gibt die Anzahl und Verteilung der Labels in allen, Trainings- und Testdaten aus."""
    print(f"\nGesamtanzahl der Stichproben: {len(df)}")
    print("\nLabelverteilung – Gesamt:")
    print(df[label_column].value_counts())
    print("\nLabelverteilung – Training:")
    print(train_df[label_column].value_counts())
    print("\nLabelverteilung – Test:")
    print(test_df[label_column].value_counts())
    print("\nLabelanteile – Training:")
    print(train_df[label_column].value_counts(normalize=True))
    print("\nLabelanteile – Test:")
    print(test_df[label_column].value_counts(normalize=True))

def plot_label_distribution(df, train_df, test_df, label_column='label'):
    """Visualisiert die Labelverteilung (Anzahl und Verhältnis) mit Balken- und Kreisdiagrammen."""
    labels_sorted = sorted(df[label_column].unique())

    def get_counts_and_ratios(data):
        return data[label_column].value_counts(), data[label_column].value_counts(normalize=True)

    all_counts, all_ratios = get_counts_and_ratios(df)
    train_counts, train_ratios = get_counts_and_ratios(train_df)
    test_counts, test_ratios = get_counts_and_ratios(test_df)

    x = range(len(labels_sorted))
    width = 0.25

    # Balkendiagramm für die Anzahl der Labels
    plt.figure(figsize=(8, 5))
    plt.bar(x, [all_counts[label] for label in labels_sorted], width=width, label='Gesamt', color='lightgray')
    plt.bar([i + width for i in x], [train_counts[label] for label in labels_sorted], width=width, label='Training', color='lightgreen')
    plt.bar([i + 2 * width for i in x], [test_counts[label] for label in labels_sorted], width=width, label='Test', color='salmon')
    plt.xticks([i + width for i in x], labels_sorted)
    plt.ylabel("Anzahl")
    plt.title("Vergleich der Labelanzahl (Gesamt vs Training vs Test)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Kreisdiagramme für die Labelanteile
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pie_colors = ['lightgray', 'lightblue', 'lightcoral']
    titles = ['Gesamtdaten', 'Trainingsdaten', 'Testdaten']

    for ax, ratios, title in zip(axes, [all_ratios, train_ratios, test_ratios], titles):
        ax.pie(
            [ratios[label] for label in labels_sorted],
            labels=labels_sorted,
            autopct='%1.1f%%',
            startangle=90,
            colors=pie_colors
        )
        ax.set_title(f"Labelanteile – {title}")

    plt.tight_layout()
    plt.show()