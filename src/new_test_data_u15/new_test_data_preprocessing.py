# data_preprocessing.py
import os
from pathlib import Path

from src.new_test_data_u15.data_cleaning import batch_clean_all_tsv
from src.new_test_data_u15.gff_validation import find_tsv_files, run_all
from src.new_test_data_u15.data_normalization_new_test_data import batch_normalize_tpm
from src.new_test_data_u15.labeling_new_test_data import process_all_files
from src.new_test_data_u15.feature_extraction import extract_features

def run_preprocessing_pipeline(
    raw_data_dir="../data/new_test_data/raw_data_test/",
    cleaned_data_dir="../data/new_test_data/cleaned_data_test/",
    normalized_data_dir="../data/new_test_data/normalized_data_test/",
    labeled_data_dir="../data/new_test_data/gene_labeling_test/",
    feature_output_path="../data/new_test_data/gene_features_test/gene_features.tsv",
    build_script_path="../src/build_features.py",
    visualize=False
):
    print("Schritt 1: Datenbereinigung")
    batch_clean_all_tsv(raw_data_dir, cleaned_data_dir, method='iqr', visual=visualize)

    print("Schritt 2: GFF-Validierung")
    tsv_files = find_tsv_files(cleaned_data_dir)
    gff_dir = os.path.join(raw_data_dir, "host_gff_files")
    run_all(tsv_files, gff_dir)

    print("Schritt 3: Normalisierung")
    batch_normalize_tpm(cleaned_data_dir, normalized_data_dir, raw_data_dir, phage_only=True, visual=visualize)

    print("Schritt 4: Label-Zuweisung")
    process_all_files(normalized_data_dir, labeled_data_dir)

    print("Schritt 5: Feature-Extraktion")
    gff_phage_dir = os.path.join(raw_data_dir, "phage_gff_files")
    fasta_dir = raw_data_dir
    labels_file = os.path.join(labeled_data_dir, "gene_labels_standard.tsv")
    Path(feature_output_path).parent.mkdir(parents=True, exist_ok=True)

    results = extract_features(
        gff_directory=gff_phage_dir,
        fasta_directory=fasta_dir,
        labels_file=labels_file,
        output_file=feature_output_path
    )

    if results is not None and not results.empty:
        print("Feature-Extraktion abgeschlossen")
        print(f"Extrahierte Gene: {len(results)}")
        print(f"Phagen: {results['source_study'].nunique()}")
        print(f"Label-Verteilung:\n{results['label'].value_counts()}")
    else:
        print("Keine Features extrahiert")

    print("Schritt 6: Feature-Matrix aufbauen")
    os.system(f"python {build_script_path}")
