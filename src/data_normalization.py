import pandas as pd
import os

from data_handling import (find_tsv_files,
                           extract_gene_lengths_from_gff,
                           find_gff_files)
from visualization import (visualize_tpm_boxplot_only,
                           visualize_tpm_boxplot_entitywise,
                           visualize_pca_comparison)


# TPM-Normalisierung mit berechneten Gen-Längen/ 1000, falls keine Gen-Länge gefunden werden kann
# data_normalization.py
def normalize_tpm(
        df: pd.DataFrame,
        count_cols,
        feature_lengths,
        entity_col: str,
        phage_only: bool = True
    ):

    # 1) RPK berechnen
    rpk = df[count_cols].div(feature_lengths / 1000, axis=0)

    # 2) Skalierungsfaktoren bestimmen
    if phage_only:
        mask = df[entity_col].eq("phage")
        scaling_factors = rpk.loc[mask].sum(axis=0) / 1e6
    else:
        scaling_factors = rpk.sum(axis=0) / 1e6

    # 3) TPM
    return rpk.divide(scaling_factors, axis=1)



def batch_normalize_tpm(input_dir, output_dir, gff_dir, phage_only=True, visual=True):
    """
    Führt TPM-Normalisierung für alle TSV-Dateien im Eingabeverzeichnis durch,
    erstellt Visualisierungen und speichert die normalisierten Daten.

    Args:
        input_dir (str): Verzeichnis mit den Eingabedateien
        output_dir (str): Verzeichnis für die Ausgabedateien
        gff_files_list (list): Liste der GFF-Dateien für die Genlängen-Extraktion
        visual (bool): Wenn True, werden Visualisierungen erstellt
    """

    gff_files_list = find_gff_files(gff_dir)

    os.makedirs(output_dir, exist_ok=True)
    tsv_files = find_tsv_files(input_dir)

    # Anwendung auf alle Dateien
    for file_path in tsv_files:
        try:
            df = pd.read_csv(file_path, sep='\t')

            gene_col = df.columns[0]
            entity_col = df.columns[-3]
            symbol_col = df.columns[-2]
            count_cols = df.columns[1:-3]

            print(f"\nVerarbeite: {os.path.basename(file_path)}")

            # Gene-Längen aus der entsprechenden GFF-Datei extrahieren
            gene_lengths = {}
            for gff_file in gff_files_list:
                gene_lengths.update(extract_gene_lengths_from_gff(gff_file))

            # Füge Gene-Length Spalte hinzu
            df['gene_length'] = df[gene_col].map(gene_lengths).fillna(1000)  # Default 1000 wenn keine Länge gefunden

            # Kopie für Originaldaten zur Visualisierung (vor Normalisierung)
            df_original = df.copy()

            # TPM-Normalisierung mit tatsächlichen Gen-Längen
            df_tpm = normalize_tpm(df, count_cols, df['gene_length'], entity_col=entity_col, phage_only=phage_only)

            # Kombiniere mit Metadaten zur Speicherung
            df_tpm[gene_col] = df[gene_col]
            df_tpm['gene_length'] = df['gene_length']
            df_tpm[entity_col] = df[entity_col]
            df_tpm[symbol_col] = df[symbol_col]

            # Spalten sortieren
            ordered_cols = [gene_col] + list(count_cols) + ['gene_length', entity_col, symbol_col]
            df_tpm = df_tpm[ordered_cols]

            # Visualisierungen nur erstellen, wenn visual=True ist
            if visual:
                # Visualisierung 1
                visualize_tpm_boxplot_only(df.copy(), df_tpm.copy(), count_cols, entity_col, file_path)

                # Visualisierung 2
                visualize_tpm_boxplot_entitywise(df_original.copy(), df_tpm.copy(), count_cols, entity_col, file_path)

                # Visualisierung 3: PCA
                visualize_pca_comparison(df_original.copy(), df_tpm.copy(), count_cols, entity_col, file_path)

            # Speichern
            out_path = os.path.join(output_dir, os.path.basename(file_path).replace('_full_raw_counts_cleaned.tsv', '_normalized_TPM.tsv'))
            df_tpm.to_csv(out_path, sep='\t', index=False)
            print(f"Gespeichert: {out_path}")

        except Exception as e:
            print(f"Fehler bei Datei {file_path}: {e}")