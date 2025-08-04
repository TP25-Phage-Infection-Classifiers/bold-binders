import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Daten laden
train_df = pd.read_csv("../data/split_dataset_us14/train_set.tsv", sep="\t")
test_df = pd.read_csv("../data/split_dataset_us14/test_set.tsv", sep="\t")

# Features und Labels extrahieren
X_train = train_df.drop(columns=["id", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["id", "label"])
y_test = test_df["label"]

# Labels codieren
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# Random Forest Modell trainieren
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train_enc)

# Vorhersage auf Testdaten
y_pred = clf.predict(X_test)

# Evaluation
report = classification_report(
    y_test_enc, y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)
accuracy = accuracy_score(y_test_enc, y_pred)

# Ausgabe als DataFrame
report_df = pd.DataFrame(report).transpose()
report_df["accuracy"] = accuracy  # Accuracy-Spalte ergänzen
print(report_df)
