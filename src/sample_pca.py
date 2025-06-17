import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np
import re
import os

# Markerformen für Replikate, inkl. "NA" für Proben ohne Replikatinformation
replicate_shapes = {
    "R1": "o", "R2": "^", "R3": "s", "NA": "X"
}

# Ausgabeordner für alle generierten PCA-Plots
# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)


def parse_sample_metadata(columns):
    """
    Extrahiert Zeitpunkte und Replikatinformation aus Spaltennamen.
    Erkennt verschiedene Formate wie 'wt-phage-35_R1', '15_R2' oder '12'.
    Gibt ein DataFrame mit 'time' und 'replicate' für jede Probe zurück.
    """
    meta = []
    for col in columns:
        # Suche nach einem Muster wie z.B. 35_R1 irgendwo im Namen
        m = re.search(r'(?P<time>\d+)[-_]?R(?P<rep>\d+)', col)
        if m:
            meta.append({
                "sample": col,
                "time": int(m.group("time")),
                "replicate": f"R{m.group('rep')}"
            })
        elif re.match(r'^\d+$', col):  # Nur Zahl, keine Replikatinformation
            meta.append({
                "sample": col,
                "time": int(col),
                "replicate": "NA"
            })
        else:
            raise ValueError(f"Unbekanntes Spaltenformat: {col}")

    return pd.DataFrame(meta).set_index("sample")


def get_combined_legend(palette):
    """
    Erstellt eine kombinierte Legende aus Zeitpunkten (Farben) und Replikaten (Formen).
    Die Zeitpunkte basieren auf einer übergebenen Farbpalette.
    Replikate ohne 'NA' werden aus der Legende ausgeschlossen.
    """
    time_legend = [
        Line2D([0], [0], marker='o', color='w', label=f"{t} min",
               markerfacecolor=color, markersize=10)
        for t, color in palette.items()
    ]
    replicate_legend = [
        Line2D([0], [0], marker=marker, color='black', label=rep,
               linestyle='None', markersize=10)
        for rep, marker in replicate_shapes.items() if rep != "NA"
    ]
    return replicate_legend + time_legend

def detect_outliers_pca(expr, meta, threshold_std=2.5):
    """
    Identifiziert Ausreißer basierend auf der PCA-Distanz zum Zentrum aller Proben.
    Proben mit einer Distanz größer als Mittelwert + threshold_std * Standardabweichung werden markiert.
    """
    df_t = expr.T
    df_scaled = StandardScaler().fit_transform(df_t)
    pca = PCA(n_components=2).fit(df_scaled)
    coords = pca.transform(df_scaled)
    df_pca = pd.DataFrame(coords, columns=["PC1", "PC2"], index=df_t.index).join(meta)
    center = df_pca[["PC1", "PC2"]].mean().values
    distances = np.linalg.norm(df_pca[["PC1", "PC2"]].values - center, axis=1)
    threshold = distances.mean() + threshold_std * distances.std()
    return df_pca.index[distances > threshold].tolist()

def run_pca_subplot(ax, expr, meta, title, exclude=None):
    """
    Führt PCA für einen gegebenen Teildatensatz aus und zeichnet das Ergebnis in ein gegebenes Subplot.
    Optional können bestimmte Proben ausgeschlossen werden.
    """
    if exclude:
        expr = expr.drop(columns=exclude, errors="ignore")
        meta = meta.drop(index=exclude, errors="ignore")

    # Daten vorbereiten und skalieren
    df_t = expr.T
    df_scaled = StandardScaler().fit_transform(df_t)
    pca = PCA(n_components=2)
    result = pca.fit_transform(df_scaled)
    df_pca = pd.DataFrame(result, columns=["PC1", "PC2"], index=df_t.index).join(meta)

    # Farbpalette für Zeitpunkte generieren
    unique_times = sorted(df_pca["time"].unique())
    cmap = plt.colormaps.get_cmap("tab10").resampled(len(unique_times))
    palette = {t: mcolors.to_hex(cmap(i)) for i, t in enumerate(unique_times)}

    # Markerformen nur verwenden, wenn mehrere Replikate vorhanden sind
    use_style = df_pca["replicate"].nunique() > 1

    style_order = list(replicate_shapes.keys())
    marker_list = list(replicate_shapes.values())
    sns.scatterplot(
        ax=ax,
        data=df_pca,
        x="PC1", y="PC2",
        hue="time",
        style="replicate" if use_style else None,
        style_order=style_order if use_style else None,
        markers=replicate_shapes if use_style else None,
        palette=palette,
        s=100
    )

    # Beschriftung neben jedem Punkt anzeigen (Replikat)
    # for _, row in df_pca.iterrows():
    #    label = row.name
    #    ax.text(row["PC1"] + 1, row["PC2"], label, fontsize=8, ha='left', va='center')


    ax.set_title(f"{title}\nPC1={pca.explained_variance_ratio_[0]*100:.2f}%, PC2={pca.explained_variance_ratio_[1]*100:.2f}%")
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.legend_.remove()
    return palette

def run_all_pcas(expr_df, gene_info, meta_df, base_name, outliers=None):
    """
    Führt vier PCA-Analysen aus:
    (a) alle Gene, (b) nur Wirtsgene, (c) nur Phagen, (d) ohne Ausreißer.
    Zeigt das Ergebnis in einer 2x2-Subplot-Matrix mit gemeinsamer Legende.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 11))
    axs = axs.flatten()

    # Auswahl der relevanten Gene
    host_genes = gene_info[gene_info["Entity"] == "host"].index
    phage_genes = gene_info[gene_info["Entity"] == "phage"].index

    # Nur automatische Ausreißererkennung, wenn mehr als 1 Replikat vorhanden ist.
    # Manuell angegebene Ausreißer werden nur berücksichtigt, wenn mehrere Replikate existieren.
    if meta_df["replicate"].nunique() > 1:
        if outliers is None:
            outliers = detect_outliers_pca(expr_df, meta_df)
    else:
        outliers = []

    plot_specs = [
        ("(a) All genes", expr_df, None),
        ("(b) Host only", expr_df.loc[expr_df.index.intersection(host_genes)], None),
        ("(c) Phage only", expr_df.loc[expr_df.index.intersection(phage_genes)], None),
        (f"(d) Outlier removed\nRemoved: {', '.join(outliers) if outliers else 'None'}", expr_df, outliers)
    ]

    last_palette = None
    for ax, (title, data, exclude) in zip(axs, plot_specs):
        last_palette = run_pca_subplot(ax, data, meta_df.copy(), title, exclude)

    # Gemeinsame Legende für alle Subplots
    fig.legend(
        get_combined_legend(last_palette),
        [h.get_label() for h in get_combined_legend(last_palette)],
        loc='lower center',
        ncol=len(last_palette) + len([r for r in replicate_shapes if r != "NA"]),
        bbox_to_anchor=(0.5, -0.03),
        frameon=True
    )
    fig.text(0.5, -0.07, f"File: {base_name}", ha='center', fontsize=10, style='italic')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    # Ausgabe des Plots statt Speichern
    plt.show()
    # plt.savefig(...) optional möglich

def process_file(file_path, user_outliers=None):
    """
    Verarbeitet eine einzelne TSV-Datei:
    - lädt die Daten,
    - entfernt Kontrollproben (Ctrl_*),
    - extrahiert Annotation und Metadaten,
    - ruft die PCA-Funktion auf.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing: {base_name}")

    df = pd.read_csv(file_path, sep='\t', index_col=0)

    annotation_cols = ["Entity", "Symbol"]
    gene_info = df[annotation_cols]

    # Nur experimentelle Spalten behalten (alle außer Ctrl_)
    expr_cols = [col for col in df.columns if not col.startswith("Ctrl_") and col not in annotation_cols + ["gene_length"]]
    expr_df = df[expr_cols]

    meta_df = parse_sample_metadata(expr_df.columns)
    expr_df = expr_df[meta_df.index].fillna(0)

    run_all_pcas(expr_df, gene_info, meta_df, base_name, outliers=user_outliers)


def process_directory(input_dir, user_outliers=None, filename_filter=None):
    """
    Durchläuft alle .tsv-Dateien im gegebenen Verzeichnis
    und führt die PCA-Verarbeitung nur für Dateien aus,
    deren Dateiname den gegebenen Filterstring (filename_filter) enthält,
    falls dieser angegeben wurde. Ansonsten werden alle Dateien verarbeitet.
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".tsv") and (filename_filter is None or filename_filter in file_name):
            process_file(os.path.join(input_dir, file_name), user_outliers)
