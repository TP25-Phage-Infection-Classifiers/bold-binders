import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from data_handling import find_tsv_files
import seaborn as sns
import matplotlib.pyplot as plt
from visualization import visualize_outliers

# Funktion zum Markieren von Ausreißern mit visual-Parameter
def mark_outliers(file_path, host_threshold, phage_threshold, method='iqr', visual=True):
    try:
        # Einlesen der TSV-Datei
        df = pd.read_csv(file_path, sep='\t')

        # Einlesen der Spaltennamen
        gene_col = df.columns[0]
        symbol_col = df.columns[-1]
        entity_col = df.columns[-2]
        count_cols = df.columns[1:-2]

        # DataFrame für Ergebnisse initialisieren
        results_df = df.copy()
        results_df['is_outlier'] = False
        results_df['outlier_info'] = ""

        # Gruppierung nach Entity (host/phage)
        host_mask = df[entity_col] == 'host'
        phage_mask = df[entity_col] == 'phage'

        # Ausreißer für Host-Gene identifizieren
        if sum(host_mask) > 0:
            host_outliers = detect_outliers(df[host_mask], count_cols, host_threshold, method, is_phage=False)
            for idx, reason in host_outliers.items():
                results_df.loc[idx, 'is_outlier'] = True
                results_df.loc[idx, 'outlier_info'] += f"Host-Outlier{reason}; "

        # Ausreißer für Phagen-Gene identifizieren (nur 0.05-Quantil-Methode)
        if sum(phage_mask) > 0:
            phage_outliers = detect_outliers(df[phage_mask], count_cols, phage_threshold, method, is_phage=True)
            for idx, reason in phage_outliers.items():
                results_df.loc[idx, 'is_outlier'] = True
                results_df.loc[idx, 'outlier_info'] += f"Phagen-Outlier{reason}; "

        # Visualisierung
        if visual:
            fig = visualize_outliers(results_df, file_path, host_mask, phage_mask)
            plt.show(fig)

        return results_df

    except Exception as e:
        print(f"Fehler {e}")
        return None


# Hilfsfunktion zur Ausreißererkennung
def detect_outliers(df, count_cols, threshold=3.0, method='iqr', is_phage=False):
    outlier_indices = {}

    # 0) log-Transform gegen Schiefe
    log_df = np.log2(df[count_cols] + 1)

    if is_phage:
        # weiterhin nur Low-Count-Filter (alles < q05)
        q05 = log_df.quantile(0.05)
        mask_low = (log_df.lt(q05, axis=1)).all(axis=1)
        for idx in df.index[mask_low]:
            outlier_indices[idx] = "(low-phage)"
        return outlier_indices

    # Host-Gene-Branch
    if method == 'mad':
        med = log_df.median()
        mad = (log_df - med).abs().median()
        z = (log_df - med) / (1.4826 * mad)
        mask = (z.abs() > threshold).any(axis=1)
        for idx in df.index[mask]:
            outlier_indices[idx] = "(mad)"
    else:              # default: IQR auf Log-Werten
        q1 = log_df.quantile(0.25)
        q3 = log_df.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (log_df.lt(lower, axis=1) | log_df.gt(upper, axis=1)).any(axis=1)
        for idx in df.index[mask]:
            outlier_indices[idx] = "(iqr-log)"
    return outlier_indices



# Funktion zur Datenbereinigung mit visual-Parameter
def clean_outlier_samples(file_path, host_threshold, phage_threshold, method='iqr', output_dir="../cleaned_data", visual=True):
    cleaned_df = mark_outliers(file_path, host_threshold=host_threshold, phage_threshold=phage_threshold, method=method, visual=visual)
    #print("danach",cleaned_df.shape())
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
def batch_clean_all_tsv(input_dir, output_dir, method='mad',visual=True):
    tsv_files = find_tsv_files(input_dir)
    host_threshold = 3
    phage_threshold = 10

    if not tsv_files:
        print(f"Keine TSV-Dateien gefunden in: {input_dir}")
        return

    print(f"{len(tsv_files)} Dateien werden verarbeitet...\n")
    for file_path in tsv_files:
        clean_outlier_samples(
            file_path=file_path,
            host_threshold=host_threshold,
            phage_threshold=phage_threshold,
            method=method,
            output_dir=output_dir,
            visual=visual  # Übergebe den visual-Parameter
        )
    print("\nAlle Dateien wurden verarbeitet.")