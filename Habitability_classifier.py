import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Loading the cleaned dataset
df = pd.read_csv("kepler_cumulative_cleaned.csv")

# Filtering rows only include confirmed
confirmedrows = df[df["koi_disposition"] == "CONFIRMED"].copy()

# This function is to label planet habitable based on koi_teq's and koi_insol's value range, based on earlier
# discussion. If planet falls within range, then it is labeled as 1, else return 0. Make a new column called
# habitability_label.
def label_habitability(row):
    if (200 <= row["koi_teq"] <= 350) and (0.3 <= row["koi_insol"] <= 1.5):
        return 1
    else:
        return 0

confirmedrows["habitability_label"] = confirmedrows.apply(label_habitability, axis=1)

# Same as before, but now without koi_score and koi_tce_plnt_num. Used fillna-median in case there is NaNs in the
# dataset. Fills them with median values. Y is targeted as 1 potential habitable, 0 not habitable within the teq
# and insol's range.
kepler_column_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_period', 'koi_duration', 'koi_depth', 'koi_impact','koi_model_snr',
    'koi_steff', 'koi_srad', 'koi_slogg'
]

X = confirmedrows[kepler_column_features].fillna(confirmedrows[kepler_column_features].median())
y = confirmedrows["habitability_label"]

# Split: same as before: 30 % test and 70 % train. 42 = Douglas Adams ;).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Same at last time. Just named rf to rf_habitalbe. Same classifier with 100 trees and seeds.
rf_habitable = RandomForestClassifier(n_estimators=100, random_state=42)
rf_habitable.fit(X_train, y_train)
y_prediction = rf_habitable.predict(X_test)

# Evaluating the model with the classification matrix (like last time).
print("Habitability Classifier (Random Forest):")
print(classification_report(y_test, y_prediction))
print("ROC AUC:", roc_auc_score(y_test, rf_habitable.predict_proba(X_test)[:, 1]))

# Showing how many were predicted as potentially habitable.
confirmedrows["predicted_habitable"] = rf_habitable.predict(X)
print("\nPotentially Habitable Planets Predicted:", confirmedrows["predicted_habitable"].sum())

# Displaying the planets
confirmedrows["habitability_prob"] = rf_habitable.predict_proba(X)[:, 1]

# First I made a top 10, 15 and 30. But figured I wanted to show the whole list.
All_54 = confirmedrows.sort_values(by="habitability_prob", ascending=False).head(54)

# Wanted to also include koi_prad in the list to see the planets size. 1.0 means Earth radius = like Earth
print("\n Potentially habitable exoplanets (by model probability):")
print(All_54[["kepoi_name", "kepler_name", "koi_prad", "koi_teq", "koi_insol", "habitability_prob"]])