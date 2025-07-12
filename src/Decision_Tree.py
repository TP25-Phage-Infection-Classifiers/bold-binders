from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# SMOTE-Strategie zur leichten Überabtastung der Klassen "late" (1) und "middle" (2)
smote_strategy = {1: 382, 2: 304}

# Finales Modell basierend auf den besten Parametern aus der GridSearchCV
# Besteht aus einer Pipeline mit SMOTE für Klassenausgleich und einem Entscheidungsbaum
final_model = Pipeline([
    ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=42)),
    ('clf', DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    ))
])
