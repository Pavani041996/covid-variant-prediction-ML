# SARS-CoV-2 Mutation Escape Prediction Model

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ---------------------------
# STEP 1: Create Dataset
# ---------------------------

data = {
    "Mutation": ["E484A", "N501Y", "L452R", "D614G", "K417N"],
    "Position": [484, 501, 452, 614, 417],
    "WT_AA": ["E", "N", "L", "D", "K"],
    "Mut_AA": ["A", "Y", "R", "G", "N"],
    "Region": ["RBD", "RBD", "RBD", "S1/S2", "RBD"],
    "Escape_Label": [1, 1, 1, 1, 1]  # High escape (example)
}

df = pd.DataFrame(data)

# ---------------------------
# STEP 2: Encode Features
# ---------------------------

le = LabelEncoder()

df["WT_AA"] = le.fit_transform(df["WT_AA"])
df["Mut_AA"] = le.fit_transform(df["Mut_AA"])
df["Region"] = le.fit_transform(df["Region"])

X = df[["Position", "WT_AA", "Mut_AA", "Region"]]
y = df["Escape_Label"]

# ---------------------------
# STEP 3: Train-Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# STEP 4: Train Model
# ---------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# STEP 5: Predictions
# ---------------------------

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# STEP 6: Feature Importance
# ---------------------------

importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance - Mutation Escape Prediction")
plt.show()
