import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/mutation_dataset.csv")

# Encode categorical features
le = LabelEncoder()

for col in ["WT_AA", "Mut_AA", "Region", "Charge_Change", "Polarity_Change", "Hydropathy_Change"]:
    df[col] = le.fit_transform(df[col])

# Features and label
X = df.drop(columns=["Mutation", "Escape_Label"])
y = df["Escape_Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance - Mutation Escape Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.show()
