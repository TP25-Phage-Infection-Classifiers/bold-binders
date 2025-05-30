import os
import pandas as pd
import matplotlib.pyplot as plt
from labeling import (
    classify_expression_with_fallback,
    classify_expression_paper_defined,
    infer_paper_key_from_filename
)
from data_handling import find_tsv_files
from visualization import (
    plot_label_distribution_comparison,
    plot_aggregated_distribution
)

def process_files_standard_method(data_folder):
    """Verarbeitet alle Dateien mit der Standard-Methode."""
    all_results = []
    tsv_files = find_tsv_files(data_folder)

    for file_path in tsv_files:
        file_name = os.path.basename(file_path)
        print(f"Verarbeite Datei mit Standard-Methode: {file_name}")
        try:
            result = classify_expression_with_fallback(file_path)
            result['SourceFile'] = file_name
            all_results.append(result)
        except Exception as e:
            print(f"Fehler bei Datei {file_name}: {e}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def process_files_paper_method(data_folder):
    """Verarbeitet alle Dateien mit der Paper-definierten Methode."""
    all_results = []
    tsv_files = find_tsv_files(data_folder)

    for file_path in tsv_files:
        file_name = os.path.basename(file_path)
        paper_key = infer_paper_key_from_filename(file_name)

        if paper_key:
            print(f"Verarbeite Datei mit Paper-Methode: {file_name} ({paper_key})")
            try:
                result = classify_expression_paper_defined(file_path, paper_key)
                result['SourceFile'] = file_name
                all_results.append(result)
            except Exception as e:
                print(f"Fehler bei Datei {file_name}: {e}")
        else:
            print(f"Kein paper_key gefunden für Datei: {file_name}")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()

def compare_labeling_methods(results_std, results_paper):
    """Vergleicht die Ergebnisse beider Klassifikationsmethoden."""
    if results_std.empty or results_paper.empty:
        print("Keine Ergebnisse zum Vergleichen verfügbar.")
        return

    merged = results_std.merge(
        results_paper,
        on=['Geneid', 'Entity', 'Symbol', 'SourceFile'],
        suffixes=('_std', '_paper')
    )

    differences = merged[merged['Label_std'] != merged['Label_paper']]

    total = len(merged)
    diff_count = len(differences)
    same_count = total - diff_count
    percentage_diff = round((diff_count / total) * 100, 2)

    print(f"Gesamtzahl gemeinsam gelabelter Gene: {total}")
    print(f"Anzahl unterschiedlich gelabelter Gene: {diff_count}")
    print(f"Anteil unterschiedlich gelabelter Gene: {percentage_diff}%")
    print(f"Anzahl gleich gelabelter Gene: {same_count}")

    return merged

def main():
    """Hauptfunktion zum Ausführen der gesamten Pipeline."""
    data_folder = "../data/normalized_data_bb/"
    output_folder = "../results/gene_labeling/"
    os.makedirs(output_folder, exist_ok=True)

    # 1. Dateien mit beiden Methoden verarbeiten
    results_std = process_files_standard_method(data_folder)
    results_paper = process_files_paper_method(data_folder)

    if results_std.empty or results_paper.empty:
        print("Fehler: Keine gültigen Ergebnisse verfügbar.")
        return

    # 2. Ergebnisse speichern
    results_std.to_csv(f"{output_folder}gene_labels_standard.csv", index=False)
    results_paper.to_csv(f"{output_folder}gene_labels_paper.csv", index=False)

    # 3. Vergleich der Methoden
    merged_results = compare_labeling_methods(results_std, results_paper)
    merged_results.to_csv(f"{output_folder}gene_labels_comparison.csv", index=False)

    # 4. Visualisierungen erstellen
    label_counts_std = results_std.groupby(['SourceFile', 'Label']).size().unstack(fill_value=0)
    label_counts_paper = results_paper.groupby(['SourceFile', 'Label']).size().unstack(fill_value=0)

    # Sortiere Spalten einheitlich
    label_order = ["early", "middle", "late"]
    label_counts_std = label_counts_std.reindex(columns=label_order)
    label_counts_paper = label_counts_paper.reindex(columns=label_order)

    # Erstelle Visualisierungen
    fig1 = plot_label_distribution_comparison(label_counts_std, label_counts_paper)
    fig1.savefig(f"{output_folder}label_distribution_comparison.png", bbox_inches='tight')

    fig2 = plot_aggregated_distribution(label_counts_std, label_counts_paper)
    fig2.savefig(f"{output_folder}aggregated_distribution.png", bbox_inches='tight')

    print(f"Alle Ergebnisse wurden im Verzeichnis '{output_folder}' gespeichert.")

if __name__ == "__main__":
    main()