import os
import pandas as pd

# Rekursive Dateisuche nach .tsv-Dateien
def find_tsv_files(directory):
    tsv_files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tsv'):
                full_path = os.path.join(root, file)
                tsv_files_list.append(full_path)
    return tsv_files_list


# Extract gene IDs from GFF file and return full DataFrame
def extract_gene_ids_from_gff(gff_file):
    with open(gff_file, 'r') as f:
        lines = [line for line in f if not line.startswith('#')]
    records = [line.strip().split('\t') for line in lines if len(line.strip().split('\t')) == 9]
    df = pd.DataFrame(records, columns=[
        'seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'
    ])
    gene_df = df[df['type'] == 'gene'].copy()
    gene_df['id_field'] = gene_df['attributes'].str.extract(r'ID=([^;]+)')
    gene_df['name_field'] = gene_df['attributes'].str.extract(r'Name=([^;]+)')
    combined_ids = pd.concat([gene_df['id_field'].dropna(), gene_df['name_field'].dropna()]).unique()
    return set(combined_ids), len(gene_df), df

def compare_with_multiple_gff_and_print_filtered(count_table_path, gff_folder_path, save_filtered=False):
    counts = pd.read_csv(count_table_path, sep='\t')
    filtered = counts[(counts['Entity'].fillna('') == '') | (counts['Entity'] == 'host')]
    filtered_gene_ids = set(filtered.iloc[:, 0])

    print(f"\nTotal 'host'/empty genes in count table: {len(filtered_gene_ids)}\n")

    for file in os.listdir(gff_folder_path):
        if file.endswith('.gff'):
            gff_path = os.path.join(gff_folder_path, file)
            gff_gene_ids, gff_total_genes, gff_df = extract_gene_ids_from_gff(gff_path)

            matched = filtered_gene_ids & gff_gene_ids
            unmatched = filtered_gene_ids - gff_gene_ids
            match_rate = len(matched) / len(filtered_gene_ids) * 100 if filtered_gene_ids else 0

            if matched:
                print(f"File: {file}")
                print(f"   Genes in GFF:               {gff_total_genes}")
                print(f"   Matched genes:               {len(matched)}")
                print(f"   Unmatched genes:             {len(unmatched)}")
                print(f"   Match rate (before removal): {match_rate:.2f}%")

                if match_rate < 100.0:
                    # Search unmatched genes in other GFF feature types
                    search_results = []
                    for gene_id in unmatched:
                        found = gff_df['attributes'].str.contains(gene_id, na=False)
                        matches = gff_df[found]
                        search_results.append({
                            'Unmatched Gene ID': gene_id,
                            'Found in any attribute': not matches.empty,
                            'Example attribute match': matches.iloc[0]['attributes'] if not matches.empty else ""
                        })
                    found_count = sum(1 for r in search_results if r['Found in any attribute'])
                    print(f"   Found in other feature types: {found_count}")
                    print(f"   First {len(unmatched)} unmatched gene IDs: {list(unmatched)[:10]}")

                    # Remove unmatched genes from the filtered DataFrame
                    before_count = len(filtered)
                    filtered = filtered[~filtered.iloc[:, 0].isin(unmatched)]
                    after_count = len(filtered)
                    print(f"   {before_count - after_count} unmatched genes removed from count table")

                    # Recalculate match rate after removal
                    new_gene_ids = set(filtered.iloc[:, 0])
                    new_matched = new_gene_ids & gff_gene_ids
                    new_match_rate = len(new_matched) / len(new_gene_ids) * 100 if new_gene_ids else 0
                    print(f"   Match rate after removal: {new_match_rate:.2f}%\n")

    if save_filtered:
        out_path = count_table_path.replace(".tsv", "_filtered.tsv")
        filtered.to_csv(out_path, sep='\t', index=False)
        print(f"\nFiltered count table saved to: {out_path}")


# Apply analysis to all count tables in a folder
def run_all(tsv_files, gff_folder_path):
    print("Comparing count tables to GFF files")
    for count_file_path in tsv_files:
        print(f"\nProcessing count table: {os.path.basename(count_file_path)}")
        compare_with_multiple_gff_and_print_filtered(count_file_path, gff_folder_path)