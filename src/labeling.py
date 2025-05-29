import pandas as pd
import re
from data_handling import find_tsv_files
import os

# Zeit-Fenster-Definitionen aus den Papers
PAPER_TIME_WINDOWS = {
    "brandao": [(0, 5, "early"), (5, 10, "middle"), (10, 15, "late")],
    "ceyssens": [(0, 5, "early"), (5, 15, "middle"), (15, 35, "late")],
    "guegler": [(2.5, 5, "early"), (5, 10, "middle"), (10, 20, "late")],
    "leskinen": [(2, 5, "early"), (10, 21, "middle"), (28, 49, "late")],
    "li": [(0, 45, "early"), (45, 75, "middle"), (75, 135, "late")],
    "sprenger": [(0, 15, "early"), (15, 60, "middle"), (60, 120, "late")],
    "wolfram-schauerte": [(0, 5, "early"), (5, 10, "middle"), (10, 20, "late")]
}

def extract_time_columns(df):
    """Extrahiert Zeitspalten aus dem DataFrame und gibt ein Dictionary mit Zeitpunkten zurück."""
    timepoint_cols = {}
    for col in df.columns:
        match_r = re.match(r'^(\d+\.?\d*)_R\d+$', col)
        match_simple = re.match(r'^(\d+\.?\d*)$', col)
        if match_r:
            time = float(match_r.group(1))
            timepoint_cols.setdefault(time, []).append(col)
        elif match_simple:
            time = float(match_simple.group(1))
            timepoint_cols.setdefault(time, []).append(col)
    return timepoint_cols

def classify_expression_with_fallback(file_path):
    """
    Klassifiziert Gene basierend auf Expressionswerten zu unterschiedlichen Zeitpunkten
    in die Phasen: 'early', 'middle' oder 'late'.
    """
    # Lese TSV-Datei ein
    df = pd.read_csv(file_path, sep='\t')

    # Nur phagen gene auswählen
    phage_genes = df[df['Entity'] == 'phage']

    # Erkenne und gruppiere Zeitspalten nach Zeitpunkten
    timepoint_cols = extract_time_columns(phage_genes)

    # Sortiere Zeitpunkte
    sorted_times = sorted(timepoint_cols.keys())
    n = len(sorted_times)

    # Fall 1: keine Zeitpunkte erkannt
    if n == 0:
        raise ValueError("Keine gültigen Zeitpunkte gefunden.")

    # Fall 2: nur ein Zeitpunkt – Klassifikation nicht möglich
    elif n == 1:
        phage_genes['Label'] = 'unknown'

    # Fall 3: zwei Zeitpunkte – teile in 'early' und 'late'
    elif n == 2:
        early, late = sorted_times
        phage_genes['mean_early'] = phage_genes[timepoint_cols[early]].mean(axis=1)
        phage_genes['mean_late'] = phage_genes[timepoint_cols[late]].mean(axis=1)
        phage_genes['Label'] = phage_genes[['mean_early', 'mean_late']].idxmax(axis=1).str.replace('mean_', '')

    # Fall 4: drei oder mehr Zeitpunkte – unterteile in Drittel
    else:
        early = sorted_times[:n // 3]
        middle = sorted_times[n // 3:2 * n // 3]
        late = sorted_times[2 * n // 3:]

        early_cols = [col for t in early for col in timepoint_cols[t]]
        middle_cols = [col for t in middle for col in timepoint_cols[t]]
        late_cols = [col for t in late for col in timepoint_cols[t]]

        phage_genes['mean_early'] = phage_genes[early_cols].mean(axis=1)
        phage_genes['mean_middle'] = phage_genes[middle_cols].mean(axis=1)
        phage_genes['mean_late'] = phage_genes[late_cols].mean(axis=1)

        # Bestimme die Phase mit der höchsten mittleren Expression
        phage_genes['Label'] = phage_genes[['mean_early', 'mean_middle', 'mean_late']].idxmax(axis=1).str.replace('mean_', '')

    # Rückgabe der wichtigsten Spalten mit Label
    return phage_genes[['Geneid', 'Entity', 'Symbol', 'Label']]

def classify_expression_paper_defined(file_path, paper_key):
    """
    Klassifiziert Gene in early/middle/late basierend auf
    paper-spezifischen Zeitfenstern (biologische Definition).
    """
    df = pd.read_csv(file_path, sep="\t")

    # Nur phagen gene auswählen
    phage_genes = df[df['Entity'] == 'phage']

    time_windows = PAPER_TIME_WINDOWS.get(paper_key)
    if time_windows is None:
        raise ValueError(f"Kein Zeitfenster für Paper '{paper_key}' definiert.")

    # Zeitspalten erkennen
    timepoint_cols = extract_time_columns(phage_genes)

    # Zeitpunkte den Phasen zuordnen
    window_columns = {"early": [], "middle": [], "late": []}
    for time, cols in timepoint_cols.items():
        for start, end, label in time_windows:
            if start <= time < end:
                window_columns[label].extend(cols)
                break

    # Mittelwert pro Phase berechnen
    for label in ["early", "middle", "late"]:
        if window_columns[label]:
            phage_genes[f"mean_{label}"] = phage_genes[window_columns[label]].mean(axis=1)

    # Bestimme Phase mit höchster Expression
    phage_genes["Label"] = phage_genes[[f"mean_{l}" for l in window_columns if window_columns[l]]].idxmax(axis=1).str.replace("mean_", "")

    return phage_genes[['Geneid', 'Entity', 'Symbol', 'Label']]

def infer_paper_key_from_filename(filename):
    """Ermittelt anhand des Dateinamens den passenden Paper-Schlüssel."""
    name = filename.lower()
    for key in PAPER_TIME_WINDOWS:
        if key in name:
            return key
    return None

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
        return None

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

def process_all_files(data_folder, output_folder):
    """Hauptfunktion zur Verarbeitung aller Dateien und Speicherung der Ergebnisse."""
    os.makedirs(output_folder, exist_ok=True)

    # 1. Dateien mit beiden Methoden verarbeiten
    results_std = process_files_standard_method(data_folder)
    results_paper = process_files_paper_method(data_folder)

    if results_std.empty or results_paper.empty:
        print("Fehler: Keine gültigen Ergebnisse verfügbar.")
        return None, None, None

    # 2. Ergebnisse speichern
    results_std.to_csv(f"{output_folder}gene_labels_standard.tsv",sep='\t', index=False)
    results_paper.to_csv(f"{output_folder}gene_labels_paper.tsv",sep='\t', index=False)

    # 3. Vergleich der Methoden
    merged_results = compare_labeling_methods(results_std, results_paper)
    if merged_results is not None:
        merged_results.to_csv(f"{output_folder}gene_labels_comparison.tsv", sep='\t', index=False)

    # Vorbereitung der Daten für Visualisierung
    label_counts_std = results_std.groupby(['SourceFile', 'Label']).size().unstack(fill_value=0)
    label_counts_paper = results_paper.groupby(['SourceFile', 'Label']).size().unstack(fill_value=0)

    # Sortiere Spalten einheitlich
    label_order = ["early", "middle", "late"]
    label_counts_std = label_counts_std.reindex(columns=label_order)
    label_counts_paper = label_counts_paper.reindex(columns=label_order)

    return label_counts_std, label_counts_paper, merged_results