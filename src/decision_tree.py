from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# SMOTE-Strategie zur leichten Überabtastung der Klassen "late" (1) und "middle" (2)
smote_strategy = {1: 382, 2: 404}

# Finales Modell basierend auf den besten Parametern aus der GridSearchCV
# Besteht aus einer Pipeline mit SMOTE für Klassenausgleich und einem Entscheidungsbaum


# better for late, but bad for middle
final_model = Pipeline([
    ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
    ('clf', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ))
])

# better for middle, but bad for late
# smote_strategy = {1: 382, 2: 304}
#final_model = Pipeline([
    #('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
    #('clf', DecisionTreeClassifier(
        #criterion='entropy',
        #max_depth=10,
        #min_samples_split=10,
        #min_samples_leaf=2,
        #max_features='sqrt',
        #random_state=42
    #))
#])


def train_and_evaluate(final_model, train_path, test_path):
    """
    Trainiert das übergebene Modell und gibt die Klassifikationsmetriken für Testdaten aus.
    """

    # Daten laden
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    X_train = train_df.drop(columns=["id", "label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["id", "label"])
    y_test = test_df["label"]

    # Label-Encoding
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Training
    final_model.fit(X_train, y_train_enc)

    # Vorhersage und Bericht
    y_pred = final_model.predict(X_test)
    print("Klassifikationsbericht (Testdaten):")
    print(classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_))
