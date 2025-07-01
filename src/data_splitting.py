import pandas as pd
from sklearn.model_selection import train_test_split

#-------------------------------splitting
# Load data
df = pd.read_csv("../data/feature_matrix/feature_matrix.tsv", sep="\t")

# Stratified sampling per label
train_parts = []
test_parts = []

for label in df["label"].unique():
    subset = df[df["label"] == label]
    train, test = train_test_split(subset, test_size=0.2, random_state=42, shuffle=True)
    train_parts.append(train)
    test_parts.append(test)

# Combine splits
train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to TSV
train_df.to_csv("../data/split_data_bb/feature_training.tsv", sep="\t", index=False)
test_df.to_csv("../data/split_data_bb/feature_test.tsv", sep="\t", index=False)

print("Stratified split complete.")