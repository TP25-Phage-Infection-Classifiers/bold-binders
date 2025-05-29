import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Funktion zur Visualisierung der Ergebnisse
def visualize_outliers(df, file_path, host_mask, phage_mask):
    # Basiseinstellungen, Verschiebung der Indices aufgrund der neuen Spalten
    gene_col = df.columns[0]
    entity_col = df.columns[-4]
    symbol_col = df.columns[-3]
    count_cols = df.columns[1:-4]

    # Mittlere Counts berechnen
    if 'mean_counts' not in df.columns:
        df['mean_counts'] = df[count_cols].mean(axis=1)

    # Erstelle eine Figure mit 1 Zeile und 2 Spalten
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle(f"Outlieranalyse: {os.path.basename(file_path)}", fontsize=16)

    # --- Vor der Outliererkennung ---

    # Scatter-Plot der mittleren Counts nach Entity (links)
    colors = {True: 'red', False: 'blue'}
    sns.scatterplot(x=df.index, y='mean_counts', hue=entity_col, data=df, ax=axs[0], alpha=0.7)
    axs[0].set_yscale('log')
    axs[0].set_title('Scatter-Plot der mittleren Counts vor Outliererkennung')
    axs[0].set_ylabel('Mittlere Counts (log)')
    axs[0].set_xlabel('Gen-Index')

    # --- Nach der Outliererkennung ---

    # Daten ohne Outlier
    df_cleaned = df[~df['is_outlier']].copy()

    # Scatter-Plot mit hervorgehobenen Outliern (rechts)
    sns.scatterplot(x=df.index, y='mean_counts', hue='is_outlier',
                    data=df, ax=axs[1], alpha=0.7,
                    palette={True: 'red', False: 'blue'})
    axs[1].set_yscale('log')
    axs[1].set_title('Scatter-Plot mit markierten Outliern')
    axs[1].set_ylabel('Mittlere Counts (log)')
    axs[1].set_xlabel('Gen-Index')
    axs[1].legend(title='Ist Outlier')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Liste der Outlier ausgeben
    outliers_df = df[df['is_outlier']].copy()

    if len(outliers_df) > 0:
        print("\n=== Liste der Outlier ===")
        print(file_path)
        print(f"Anzahl der Outlier: {len(outliers_df)}")
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


import matplotlib.pyplot as plt
import pandas as pd


def extract_paper_short_name(filename):
    """Ermittelt anhand des Dateinamens den Paper-Kurznamen."""
    papers = [
        "brandao", "ceyssens", "guegler", "leskinen",
        "li", "sprenger", "wolfram-schauerte"
    ]
    filename_lower = filename.lower()
    for paper in papers:
        if paper in filename_lower:
            return paper.capitalize()
    return filename


def plot_label_distribution_comparison(label_counts_std, label_counts_paper):
    """Erstellt Balkendiagramme zum Vergleich der Klassifikationsmethoden."""
    label_order = ["early", "middle", "late"]
    colors = {"early": "green", "middle": "blue", "late": "red"}
    color_list = [colors[label] for label in label_order]

    # Sortiere Dateien nach Anzahl gelabelter Gene
    ordered_files = label_counts_std.sum(axis=1).sort_values(ascending=False).index
    label_counts_std = label_counts_std.loc[ordered_files]
    label_counts_paper = label_counts_paper.reindex(ordered_files).fillna(0)

    # Kürze die Dateinamen für X-Achse
    short_labels = [extract_paper_short_name(f) for f in ordered_files]

    # Erstelle zwei nebeneinanderliegende Balkendiagramme
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Plot 1: Klassifikation mit Standard-Methode
    label_counts_std.plot(
        kind='bar',
        stacked=True,
        color=color_list,
        ax=axes[0],
        legend=False
    )
    axes[0].set_title("Standard-Methode (Drittelung)")
    axes[0].set_ylabel("Anzahl Gene")
    axes[0].set_xticks(range(len(short_labels)))
    axes[0].set_xticklabels(short_labels, rotation=45, fontsize=10)

    # Plot 2: Klassifikation mit paper-definierter Methode
    label_counts_paper.plot(
        kind='bar',
        stacked=True,
        color=color_list,
        ax=axes[1],
        legend=False
    )
    axes[1].set_title("Paper-definierte Methode")
    axes[1].set_xticks(range(len(short_labels)))
    axes[1].set_xticklabels(short_labels, rotation=45, fontsize=10)

    # Gemeinsame Legende
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Label", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.subplots_adjust(right=0.85)
    return fig


def plot_aggregated_distribution(label_counts_std, label_counts_paper):
    """Erstellt Kreis- und Balkendiagramme der Gesamtverteilung."""
    label_order = ["early", "middle", "late"]
    colors = {"early": "green", "middle": "blue", "late": "red"}
    color_list = [colors[label] for label in label_order]

    total_std = label_counts_std.sum()
    total_paper = label_counts_paper.sum()

    percent_std = total_std / total_std.sum() * 100
    percent_paper = total_paper / total_paper.sum() * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Kreisdiagramme
    axes[0, 0].pie(
        total_std,
        autopct='%1.1f%%',
        colors=color_list,
        startangle=90,
        textprops=dict(fontsize=20),
        pctdistance=1.15
    )
    axes[0, 0].set_title(
        f"Gesamtverteilung Standard-Methode\n(n={total_std.sum()} Gene)",
        fontsize=16
    )
    axes[0, 0].legend(label_order, loc="best", fontsize=12)

    axes[0, 1].pie(
        total_paper,
        autopct='%1.1f%%',
        colors=color_list,
        startangle=90,
        textprops=dict(fontsize=20)
    )
    axes[0, 1].set_title(
        f"Gesamtverteilung Paper-definierte Methode\n(n={total_paper.sum()} Gene)",
        fontsize=16
    )
    axes[0, 1].legend(label_order, loc="best", fontsize=12)

    # Balkendiagramme
    axes[1, 0].bar(label_order, percent_std, color=color_list)
    axes[1, 0].set_title(
        f"Gesamtverhältnis Standard-Methode\n(n={total_std.sum()} Gene)",
        fontsize=16
    )
    axes[1, 0].set_ylabel("Prozent", fontsize=14)
    axes[1, 0].set_ylim(0, 100)
    for i, v in enumerate(percent_std):
        axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=20)

    axes[1, 1].bar(label_order, percent_paper, color=color_list)
    axes[1, 1].set_title(
        f"Gesamtverhältnis Paper-definierte Methode\n(n={total_paper.sum()} Gene)",
        fontsize=16
    )
    axes[1, 1].set_ylabel("Prozent", fontsize=14)
    axes[1, 1].set_ylim(0, 100)
    for i, v in enumerate(percent_paper):
        axes[1, 1].text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=20)

    plt.tight_layout()
    return fig