import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Funktion zur Visualisierung der Ergebnisse
def visualize_cleaned(df, file_path, host_mask, phage_mask):
    # Basiseinstellungen, Verschiebung der Indices aufgrund der neuen Spalten
    gene_col = df.columns[0]
    entity_col = df.columns[-4]
    symbol_col = df.columns[-3]
    count_cols = df.columns[1:-4]

    # Mittlere Counts berechnen
    if 'mean_counts' not in df.columns:
        df['mean_counts'] = df[count_cols].mean(axis=1)

    # Erstelle eine Figure mit 2 Zeilen und 2 Spalten
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle(f"Ausreißeranalyse: {os.path.basename(file_path)}", fontsize=16)

    # --- Vor der Ausreißererkennung ---

    # Boxplot der mittleren Counts nach Entity (oben links)
    sns.boxplot(x=entity_col, y='mean_counts', data=df, ax=axs[0, 0])
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Boxplot der mittleren Counts vor Ausreißererkennung')
    axs[0, 0].set_ylabel('Mittlere Counts (log)')

    # Scatter-Plot der mittleren Counts nach Entity (oben rechts)
    colors = {True: 'red', False: 'blue'}
    sns.scatterplot(x=df.index, y='mean_counts', hue=entity_col, data=df, ax=axs[0, 1], alpha=0.7)
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Scatter-Plot der mittleren Counts vor Ausreißererkennung')
    axs[0, 1].set_ylabel('Mittlere Counts (log)')
    axs[0, 1].set_xlabel('Gen-Index')

    # --- Nach der Ausreißererkennung ---

    # Daten ohne Ausreißer
    df_cleaned = df[~df['is_outlier']].copy()

    # Boxplot ohne Ausreißer (unten links)
    sns.boxplot(x=entity_col, y='mean_counts', data=df_cleaned, ax=axs[1, 0])
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Boxplot der mittleren Counts nach Ausreißererkennung')
    axs[1, 0].set_ylabel('Mittlere Counts (log)')

    # Scatter-Plot mit hervorgehobenen Ausreißern (unten rechts)
    sns.scatterplot(x=df.index, y='mean_counts', hue='is_outlier',
                    data=df, ax=axs[1, 1], alpha=0.7,
                    palette={True: 'red', False: 'blue'})
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Scatter-Plot mit markierten Ausreißern')
    axs[1, 1].set_ylabel('Mittlere Counts (log)')
    axs[1, 1].set_xlabel('Gen-Index')
    axs[1, 1].legend(title='Ist Ausreißer')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Liste der Ausreißer ausgeben
    outliers_df = df[df['is_outlier']].copy()

    if len(outliers_df) > 0:
        print("\n=== Liste der Ausreißer ===")
        print(file_path)
        print(f"Anzahl der Ausreißer: {len(outliers_df)}")
        print(
            f"{'Index':<7} | {'Gen-ID':<30} | {'Entity':<8} | {'Symbol':<15} | {'Mittlere Counts':>15} | Outlier-Info")
        print("-" * 100)

        for idx, row in outliers_df.iterrows():
            gene_id = str(row[gene_col])[:28] + '..' if len(str(row[gene_col])) > 30 else str(row[gene_col])
            entity = row[entity_col]
            symbol = str(row[symbol_col])[:13] + '..' if len(str(row[symbol_col])) > 15 else str(row[symbol_col])
            mean_count = row['mean_counts']
            outlier_info = row['outlier_info']

            print(f"{idx:<7} | {gene_id:<30} | {entity:<8} | {symbol:<15} | {mean_count:>15.2f} | {outlier_info}")

    return fig


# Visualisierung TPM-Vergleich

# Boxplots Vorher-Nachher
# Visualisierung: Boxplot TPM vorher/nachher getrennt
def visualize_tpm_boxplot_only(df_before, df_after, count_cols, entity_col, file_path):
    df_before['mean_counts'] = df_before[count_cols].mean(axis=1)
    df_after['mean_tpm'] = df_after[count_cols].mean(axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_before, x=entity_col, y='mean_counts', ax=axs[0])
    axs[0].set_yscale('log')
    axs[0].set_title('Boxplot (log) vor Normalisierung')

    sns.boxplot(data=df_after, x=entity_col, y='mean_tpm', ax=axs[1])
    axs[1].set_yscale('log')
    axs[1].set_title('Boxplot (log) nach TPM')

    plt.suptitle(f"TPM-Vergleich: {os.path.basename(file_path)}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Visualisierung 1: Boxplot TPM vorher/nachher getrennt nach Entity
def visualize_tpm_boxplot_only(df_before, df_after, count_cols, entity_col, file_path):
    df_before['mean_counts'] = df_before[count_cols].mean(axis=1)
    df_after['mean_tpm'] = df_after[count_cols].mean(axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_before, x=entity_col, y='mean_counts', ax=axs[0])
    axs[0].set_yscale('log')
    axs[0].set_title('Boxplot (log) vor Normalisierung')

    sns.boxplot(data=df_after, x=entity_col, y='mean_tpm', ax=axs[1])
    axs[1].set_yscale('log')
    axs[1].set_title('Boxplot (log) nach TPM')

    plt.suptitle(f"TPM-Vergleich: {os.path.basename(file_path)}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Visualisierung 2:host/phage jeweils Before vs After
def visualize_tpm_boxplot_entitywise(df_before, df_after, count_cols, entity_col, file_path):
    # Copy and compute mean values
    df_before = df_before.copy()
    df_after = df_after.copy()
    df_before['mean'] = df_before[count_cols].mean(axis=1)
    df_after['mean'] = df_after[count_cols].mean(axis=1)
    df_before['Normalization'] = 'Before'
    df_after['Normalization'] = 'After'

    # Combine for both entities separately
    combined = pd.concat([df_before, df_after], ignore_index=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    for i, entity in enumerate(['host', 'phage']):
        subset = combined[combined[entity_col] == entity]
        if subset.empty:
            print(f"No data found for {entity}")
            continue
        sns.boxplot(x='Normalization', y='mean', data=subset, ax=axs[i])
        axs[i].set_yscale('log')
        axs[i].set_title(f"{entity.capitalize()} – TPM Before vs After")
        axs[i].set_ylabel("Mean Expression (log scale)")
        axs[i].set_xlabel("")

    plt.suptitle(f"TPM Normalization Comparison: {os.path.basename(file_path)}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Visualisierung 3: PCA-Vergleich vor und nach TPM-Normalisierung (host vs phage farblich unterschieden)
def visualize_pca_comparison(df_before, df_after, count_cols, entity_col, file_path):
    # NaN-Werte durch 0 ersetzen
    X_before = df_before[count_cols].fillna(0)
    X_after = df_after[count_cols].fillna(0)
    labels = df_before[entity_col]

    # Standardisierung der Daten
    scaler = StandardScaler()
    X_before_scaled = scaler.fit_transform(X_before)
    X_after_scaled = scaler.transform(X_after)

    # PCA auf den "before"-Datensatz fitten und beide transformieren
    pca = PCA(n_components=2)
    pc_before = pca.fit_transform(X_before_scaled)
    pc_after = pca.transform(X_after_scaled)

    # DataFrames für die PCA-Ergebnisse erstellen
    pc_df_before = pd.DataFrame(pc_before, columns=['PC1', 'PC2'])
    pc_df_before[entity_col] = labels.values

    pc_df_after = pd.DataFrame(pc_after, columns=['PC1', 'PC2'])
    pc_df_after[entity_col] = labels.values

    # Diagramme vorbereiten
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=pc_df_before, x='PC1', y='PC2', hue=entity_col,
                    palette={'host': 'blue', 'phage': 'green'}, ax=axs[0])
    axs[0].set_title('PCA vor TPM')
    axs[0].set_xlabel('PC1')
    axs[0].set_ylabel('PC2')

    sns.scatterplot(data=pc_df_after, x='PC1', y='PC2', hue=entity_col,
                    palette={'host': 'blue', 'phage': 'green'}, ax=axs[1])
    axs[1].set_title('PCA nach TPM')
    axs[1].set_xlabel('PC1')
    axs[1].set_ylabel('PC2')

    # Gleiche Achsenbereiche setzen für bessere Vergleichbarkeit
    xmin = min(pc_df_before['PC1'].min(), pc_df_after['PC1'].min())
    xmax = max(pc_df_before['PC1'].max(), pc_df_after['PC1'].max())
    ymin = min(pc_df_before['PC2'].min(), pc_df_after['PC2'].min())
    ymax = max(pc_df_before['PC2'].max(), pc_df_after['PC2'].max())

    axs[0].set_xlim(xmin, xmax)
    axs[1].set_xlim(xmin, xmax)
    axs[0].set_ylim(ymin, ymax)
    axs[1].set_ylim(ymin, ymax)

    # Gesamttitel hinzufügen
    plt.suptitle(f'PCA Vergleich – {os.path.basename(file_path)}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()