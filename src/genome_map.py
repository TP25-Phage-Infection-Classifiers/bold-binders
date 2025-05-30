import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Extrahiere Gene-IDs, Start- und Endpositionen aus einer GFF-Datei
def parse_gff_gene_ids(gff_path):
    with open(gff_path, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
    records = [line.strip().split('\t') for line in lines if len(line.strip().split('\t')) == 9]
    df = pd.DataFrame(records, columns=[
        'seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'
    ])
    df = df[df['type'] == 'gene']
    df['Geneid'] = df['attributes'].str.extract(r'(?:ID|Name)=([^;]+)')
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    return df[['Geneid', 'start', 'end']]

# Bestimme die Genomlänge aus einer GFF-Datei
def get_genome_length_from_gff(gff_path):
    with open(gff_path, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
    records = [line.strip().split('\t') for line in lines if len(line.strip().split('\t')) == 9]
    df = pd.DataFrame(records, columns=[
        'seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'
    ])
    df['end'] = df['end'].astype(int)
    return df['end'].max()

# Durchsuche ein Verzeichnis nach allen GFF/GFF3-Dateien
def find_gff_files(directory):
    gff_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gff') or file.endswith('.gff3'):
                full_path = os.path.join(root, file)
                gff_files_list.append(full_path)
    return gff_files_list

# Zeichne eine Genomkarte für die gegebenen Gene basierend auf der GFF-Datei
def plot_genome_map(tsv_df, gff_path, output_path, genome_length):
    gene_df = parse_gff_gene_ids(gff_path)
    merged = pd.merge(tsv_df, gene_df, on='Geneid', how='inner')

    colors = {"early": "green", "middle": "blue", "late": "red"}

    fig, ax = plt.subplots(figsize=(12, 2))
    for _, row in merged.iterrows():
        ax.plot([row['start'], row['end']], [1, 1], lw=10, color=colors.get(row.get('Label_std', ''), 'gray'))
        #ax.text((row['start'] + row['end']) / 2, 1.05, row['Geneid'], ha='center', fontsize=6, rotation=45)

    ax.set_xlim(0, genome_length + 200)
    ax.set_yticks([])
    ax.set_xlabel("Genomposition (bp)")
    ax.set_title("", pad=20)

    legend = [mpatches.Patch(color=v, label=k.capitalize()) for k, v in colors.items()]
    ax.legend(handles=legend, loc='upper right')

    plt.tight_layout()
    #plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    #plt.close()


from IPython.display import display
# Hauptfunktion zur Zuordnung von GFF-Dateien basierend auf GeneIDs und zur Visualisierung
def visualize_from_merged_tsv_by_geneid(merged_tsv_path, gff_dir, output_dir):
    # Erstelle das Ausgabe-Verzeichnis, falls es nicht existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lade die annotierte TSV-Datei
    full_df = pd.read_csv(merged_tsv_path, sep='\t')
    
    # Suche alle GFF-Dateien im angegebenen Verzeichnis
    gff_files = find_gff_files(gff_dir)

    # Iteriere über jede Gruppe basierend auf der Spalte 'SourceFile'
    for sourcefile, group_df in full_df.groupby('SourceFile'):
        geneids = set(group_df['Geneid'])
        matched_gff = None

        # Finde die passende GFF-Datei, die alle Gene abdeckt
        for gff_file in gff_files:
            gff_geneids = parse_gff_gene_ids(gff_file)['Geneid']
            if geneids.issubset(set(gff_geneids)):
                matched_gff = gff_file
                break

        # Wenn keine passende GFF-Datei gefunden wurde, überspringen
        if not matched_gff:
            print(f"[Warnung] Keine passende GFF-Datei für '{sourcefile}' gefunden.")
            continue

        try:
            # Bestimme die Genomlänge aus der GFF-Datei
            genome_length = get_genome_length_from_gff(matched_gff)

            # Erstelle Pfad für die Ausgabedatei
            output_file = os.path.join(
                output_dir, f"{os.path.splitext(sourcefile)[0]}_genomkarte.png"
            )

            print(f"[Verarbeite] Visualisierung von: {sourcefile} mit {os.path.basename(matched_gff)}")

            # Zeichne die Genomkarte und erhalte das Figure-Objekt
            plot_genome_map(group_df, matched_gff, output_file, genome_length)

            # Speichere das Bild als PNG-Datei
            #fig.savefig(output_file, bbox_inches='tight')
            #print(f"[Gespeichert] → {output_file}")

        except Exception as e:
            print(f"[Fehler] bei {sourcefile}: {e}")

