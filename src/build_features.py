import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils import gc_fraction
from itertools import product
from collections import Counter

# ================================
# Parameter
# ================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
NUCLEOTIDES = "ACGT"
KMER_SIZE = 3
INPUT_PATH = "../data/gene_features/gene_features.tsv"
OUTPUT_PATH = "../data/feature_matrix/feature_matrix.tsv"

# ================================
# Hilfsfunktionen
# ================================

def all_kmers(alphabet, k):
    return [''.join(p) for p in product(alphabet, repeat=k)]

def count_kmers(seq, k, alphabet, prefix):
    kmers = all_kmers(alphabet, k)
    counts = {f"{prefix}_kmer_{k}_{kmer}": 0 for kmer in kmers}
    kmers_in_seq = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    total = len(kmers_in_seq)
    counter = Counter(kmers_in_seq)
    for kmer, v in counter.items():
        if f"{prefix}_kmer_{k}_{kmer}" in counts:
            counts[f"{prefix}_kmer_{k}_{kmer}"] = v / total if total > 0 else 0
    return counts

def count_codons(dna_seq):
    return {
        "start_codon_count": dna_seq.count("ATG"),
        "stop_codon_count": sum(dna_seq.count(codon) for codon in ["TAA", "TAG", "TGA"])
    }

def count_dipeptides(prot_seq):
    dipeptides = all_kmers(AMINO_ACIDS, 2)
    counts = {f"dipep_{dp}": 0 for dp in dipeptides}
    for i in range(len(prot_seq) - 1):
        dp = prot_seq[i:i+2]
        if dp in counts:
            counts[f"dipep_{dp}"] += 1
    total = len(prot_seq) - 1
    return {k: v / total if total > 0 else 0 for k, v in counts.items()}

def clean_protein_sequence(seq):
    return ''.join([aa for aa in seq if aa in AMINO_ACIDS])

# ================================
# Daten einlesen
# ================================
df = pd.read_csv(INPUT_PATH, sep="\t")

features = []

# ================================
# Feature-Extraktion
# ================================
for _, row in df.iterrows():
    seq_id = row['gene_id']
    dna_seq = str(row['dna_sequence']).upper()
    prot_raw = str(row['amino_acid_sequence']).strip("*").upper()  # nur terminales Stoppsignal entfernen
    prot_seq = clean_protein_sequence(prot_raw)

    # Skip falls zu kurz
    if len(dna_seq) < KMER_SIZE or len(prot_seq) < 10:
        print(f"⚠️ Überspringe {seq_id} (zu kurz oder ungültig)")
        continue

    # DNA-Features
    dna_feats = {
        "id": seq_id,
        "label": row.get("label", None),
        "dna_length": len(dna_seq),
        "gc_content": gc_fraction(dna_seq),
    }
    dna_feats.update(count_kmers(dna_seq, KMER_SIZE, NUCLEOTIDES, prefix="dna"))
    dna_feats.update(count_codons(dna_seq))

    # Protein-Features
    try:
        pan = ProteinAnalysis(prot_seq)
        prot_feats = {
            "protein_length": len(prot_seq),
            "gravy": pan.gravy(),
            "isoelectric_point": pan.isoelectric_point(),
            "instability_index": pan.instability_index(),
        }
        prot_feats.update({f"aa_{aa}": v for aa, v in pan.amino_acids_percent.items()})
        prot_feats.update(count_dipeptides(prot_seq))
        prot_feats.update(count_kmers(prot_seq, KMER_SIZE, AMINO_ACIDS, prefix="prot"))

        combined_feats = {**dna_feats, **prot_feats}
        features.append(combined_feats)

    except Exception as e:
        print(f"Fehler bei {seq_id}: {e}")

# ================================
# Speichern
# ================================
df_features = pd.DataFrame(features)
df_features.to_csv(OUTPUT_PATH,sep='\t', index=False)
print(f"Feature-Matrix gespeichert unter: {OUTPUT_PATH}")
