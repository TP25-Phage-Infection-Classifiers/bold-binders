import pandas as pd
import re

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