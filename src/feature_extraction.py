import pandas as pd
import os
from Bio import SeqIO
from Bio.Seq import Seq
from pathlib import Path
from data_handling import find_gff_files, find_tsv_files, find_fasta_files, extract_gene_pos_with_id, extract_direction, extract_reading_frame


def extract_features(gff_directory, fasta_directory, labels_file, output_file):
    """
    Extrahiert Gen-Features aus Phagen und speichert sie in einer TSV-Datei.

    Args:
        gff_directory (str): Pfad zum Verzeichnis mit GFF-Dateien
        fasta_directory (str): Pfad zum Verzeichnis mit FASTA-Dateien
        labels_file (str): Pfad zur Datei mit den Gen-Labels und GeneIDs
        output_file (str): Pfad zur Ausgabe-TSV-Datei
    """
    try:
        # Labels einlesen
        labels_df = pd.read_csv(labels_file, sep='\t')
        print(f"Labels geladen: {len(labels_df)} Einträge")

        # GFF- und FASTA-Dateien finden
        gff_files = find_gff_files(gff_directory)
        fasta_files = find_fasta_files(fasta_directory)

        print(f"Gefunden: {len(gff_files)} GFF-Dateien, {len(fasta_files)} FASTA-Dateien")

        # Liste für alle Ergebnisse
        all_results = []

        # Für jede Zeile im Labels DataFrame
        for idx, row in labels_df.iterrows():
            gene_id = row.iloc[0]  # Erste Spalte ist Gene-ID
            gene_label = row.iloc[3] if len(row) > 3 else None  # Label-Spalte
            gene_source = row.iloc[4] if len(row) > 4 else None  # Source-Spalte

            # Suche in allen GFF-Dateien nach dem Gen
            for gff_file in gff_files:
                start_pos, end_pos = extract_gene_pos_with_id(gene_id, gff_file)
                direction = extract_direction(gene_id, gff_file)
                reading_frame = extract_reading_frame(gene_id, gff_file)

                if start_pos is not None and end_pos is not None and direction in ["+", "-"] and reading_frame in [0,1,2]:
                    phage_name = Path(gff_file).stem
                    matching_fasta = [f for f in fasta_files if Path(f).stem == phage_name]

                    if not matching_fasta:
                        print(f"Warnung: Keine FASTA-Datei gefunden für {phage_name}")
                        continue

                    try:
                        # FASTA-Sequenz laden
                        with open(matching_fasta[0]) as f:
                            sequence = str(next(SeqIO.parse(f, "fasta")).seq)

                            # Gen-Sequenz extrahieren (start_pos ist 1-basiert)
                            gene_sequence = sequence[start_pos - 1:end_pos]
                            biopython_seq = Seq(gene_sequence)
                            if direction == "-":
                                biopython_seq = biopython_seq.reverse_complement()
                            biopython_seq = str(biopython_seq.translate())

                            all_results.append({
                                'gene_id': gene_id,
                                'label': gene_label,
                                'dna_sequence': gene_sequence,
                                'amino_acid_sequence': biopython_seq,
                                'source_study': gene_source,
                                'phage_name': phage_name
                            })

                            print(f"Gen {gene_id} in {phage_name} gefunden und extrahiert")
                            break  # Gen gefunden, weiter zum nächsten
                    except Exception as e:
                        print(f"Fehler bei der Verarbeitung von {matching_fasta[0]}: {e}")

        # Ergebnisse in DataFrame konvertieren
        results_df = pd.DataFrame(all_results)

        if not results_df.empty:
            # TSV-Datei speichern
            results_df.to_csv(output_file, sep='\t', index=False)
            print(f"Extrahierte Features wurden in {output_file} gespeichert")
            print(f"Insgesamt {len(results_df)} Gene verarbeitet")
        else:
            print("Keine Gene wurden extrahiert!")

        return results_df

    except Exception as e:
        print(f"Fehler bei der Feature-Extraktion: {str(e)}")
        return pd.DataFrame()  # Leerer DataFrame im Fehlerfall


