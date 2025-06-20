import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the cleaned dataset
df = pd.read_csv("kepler_cumulative_cleaned.csv")

# Filtering rows to only include confirmed and the false positive
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# Numerically encode the column with 1 and 0 and creating a new column (label) in the df.
df['label'] = df['koi_disposition'].apply(lambda x: 1 if x == 'CONFIRMED' else 0)

# The columns/features are selected for imput features, same as in the heatmap (figure 33) for the same reason.
# Adding some more important columns - koi_score and koi_tce_plnt_num. X holds df with the columns and is my feature
# matrix. Y=target variable. After some test running, including all of these gave the best result.
kepler_column_features = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol',
    'koi_model_snr', 'koi_score', 'koi_tce_plnt_num', 'koi_impact',
    'koi_steff', 'koi_slogg', 'koi_srad'
]
X = df[kepler_column_features]
y = df['label']

# Handle missing values in case there is.
X = X.fillna(X.median())

# Nornalising values so that they all have a mean=0, std=1
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html., Log. reg.
# behaves weird without.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset to training and testing. 30 % test and 70 % train. Random state controls random shuffling, ensures
# spit is the same every time we run it. Of course I use 42 in this code since it represent something very applicable
# to this project - Answer to the ultimate question of life, universe and everything!! - (Douglas Adams)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Training the log. reg. model. Setting it to 1000 max to find best coefficient. Predicts class label 0 or 1.
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Training random forest model for comparison
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Must evaluate and compare the two models. Classification metrics: Precision, recall, f1 score etc. Return the
# predicted probability class 1.
print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
print()

print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# Wanted to plot a confusion matrix to show true labels vs. predicted labels. "d" formating the annotations as
# integers.
def confirmed_matrix_plotting(model_name, true_labels, predicted_labels):
    matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Purples")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

confirmed_matrix_plotting("Logistic Regression", y_test, y_pred_lr)
confirmed_matrix_plotting("Random Forest", y_test, y_pred_rf)