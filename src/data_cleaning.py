import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from data_handling import find_tsv_files
import seaborn as sns
import matplotlib.pyplot as plt
from visualization import visualize_outliers

# Funktion zum Markieren von Ausreißern mit visual-Parameter
def mark_outliers(file_path, method='iqr', threshold=1.5, visual=True):

    try:
        # Einlesen der TSV-Datei
        df = pd.read_csv(file_path, sep='\t')

        # Einlesen der Spaltennamen
        gene_col = df.columns[0] # erste Spalte ist immer das Gen
        symbol_col = df.columns[-1] # letzte Spalte ist immer das Symbol
        entity_col = df.columns[-2] # vorletzte Spalte ist immer die Entity

        # Count-Spalten extrahieren
        count_cols = df.columns[1:-2]

        # DataFrame für Ergebnisse initialisieren
        results_df = df.copy()
        results_df['is_outlier'] = False
        results_df['outlier_info']  = ""

        # Gruppierung nach Entity (host/phage)
        host_mask = df[entity_col] == 'host'
        phage_mask = df[entity_col] == 'phage'

        # Anzahl der Host- und Phagen-Gene
        #print(f"Identifizierte Gene: {sum(host_mask)} Host-Gene, {sum(phage_mask)} Phagen-Gene")

        # Ausreißer für Host-Gene identifizieren
        if sum(host_mask) > 0:
            host_outliers = detect_outliers(df[host_mask], count_cols, method, threshold)
            for idx in host_outliers:
                results_df.loc[idx, 'is_outlier'] = True
                results_df.loc[idx, 'outlier_info'] += "Host-Outlier; "

        # Ausreißer für Phagen-Gene identifizieren
        if sum(phage_mask) > 0:
            phage_outliers = detect_outliers(df[phage_mask], count_cols, method, threshold)
            for idx in phage_outliers:
                results_df.loc[idx, 'is_outlier'] = True
                results_df.loc[idx, 'outlier_info'] += "Phagen-Outlier; "

        # Visualisierung nur erstellen, wenn visual=True ist
        if visual:
            fig = visualize_outliers(results_df, file_path, host_mask, phage_mask)
            plt.show(fig)

        return results_df

    except Exception as e:
        print(f"Fehler {e}")
        return None

# Hilfsfunktion zur Ausreißererkennung
def detect_outliers(df, count_cols, method='iqr', threshold=1.5):
    outlier_indices = set()

    if method == 'iqr':
        # Interquartilsabstand-Methode
        for col in count_cols:
            Q05 = df[col].quantile(0.05)
            Q1 = df[col].quantile(0.25)
            #print(f"Q1: {Q1}")
            Q3 = df[col].quantile(0.75)
            #print(f"Q3: {Q3}")
            IQR = Q3 - Q1
            #print(f"IQR: {IQR}")
            lower_bound = Q1 - threshold * IQR
            #print(f"Lower Bound: {lower_bound}")
            upper_bound = Q3 + threshold * IQR
            #print(f"Upper Bound: {upper_bound}")
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.update(outliers)

         # Berechne das 0.05-Quantil für jede Spalte
        quantiles_05 = {col: df[col].quantile(0.05) for col in count_cols}

        # Prüfe für jede Zeile, ob alle Werte kleiner als das 0.05-Quantil sind
        for idx, row in df.iterrows():
            if all(row[col] < quantiles_05[col] for col in count_cols):
                outlier_indices.add(idx)

    elif method == 'zscore':
        # Z-Score-Methode
        for col in count_cols:
            z_scores = stats.zscore(df[col], nan_policy='omit')
            outliers = df[abs(z_scores) > threshold].index
            outlier_indices.update(outliers)

    return outlier_indices

# Funktion zur Datenbereinigung mit visual-Parameter
def clean_outlier_samples(file_path, method='iqr', threshold=1.5, output_dir="../cleaned_data", visual=True):
    cleaned_df = mark_outliers(file_path, method=method, threshold=threshold, visual=visual)

    if cleaned_df is None:
        print(f"Fehler beim Verarbeiten der Datei: {file_path}")
        return

    # Count the number of outliers
    num_outliers = cleaned_df["is_outlier"].sum()

    # Remove outliers and drop related columns
    cleaned_df = cleaned_df[~cleaned_df["is_outlier"]].copy()
    cleaned_df.drop(columns=["is_outlier", "outlier_info"], inplace=True, errors='ignore')

    # Save the cleaned file
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(file_path).replace("_full_raw_counts.tsv", "_cleaned.tsv")
    save_path = os.path.join(output_dir, file_name)
    cleaned_df.to_csv(save_path, sep='\t', index=False)

    # Output result summary
    print(f"{num_outliers} Ausreißer wurden entfernt → {file_name}")
    print(f"Finalisierte Datei gespeichert: {save_path}")

    return cleaned_df

# Batch-Funktion mit visual-Parameter
def batch_clean_all_tsv(input_dir, output_dir, method='iqr', threshold=1.5, visual=True):
    tsv_files = find_tsv_files(input_dir)

    if not tsv_files:
        print(f"Keine TSV-Dateien gefunden in: {input_dir}")
        return

    print(f"{len(tsv_files)} Dateien werden verarbeitet...\n")
    for file_path in tsv_files:
        clean_outlier_samples(
            file_path=file_path,
            method=method,
            threshold=threshold,
            output_dir=output_dir,
            visual=visual  # Übergebe den visual-Parameter
        )
    print("\nAlle Dateien wurden verarbeitet.")