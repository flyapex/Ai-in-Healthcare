"""
Comparative Analysis of ML Classifiers for Medical Diagnosis
Dataset: Pima Indians Diabetes Database
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# ==========================================
# PHASE A: Data Engineering
# ==========================================
print("--- PHASE A: Data Engineering ---")

# FIX: Specify the filename inside the dataset
file_path = "diabetes.csv"

# Updated to dataset_load() to resolve DeprecationWarning
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "uciml/pima-indians-diabetes-database",
    file_path,
)

print("First 5 records:\n", df.head())

# 1. Preprocessing: Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# 2. Correlation Matrix: Top 5 features most correlated with 'Outcome'
corr_matrix = df.corr()
top_5_features = (
    corr_matrix["Outcome"].abs().drop("Outcome").sort_values(ascending=False).head(5)
)
print("\nTop 5 features most correlated with Outcome:\n", top_5_features)

# 3. Feature Scaling
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# ==========================================
# PHASE B: Model Implementation
# ==========================================
print("\n--- PHASE B: Model Implementation ---")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

results = {"Accuracy": [], "Precision": [], "Recall": []}
model_names = list(models.keys())
best_model_name = ""
highest_accuracy = 0
best_y_pred = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results["Accuracy"].append(acc)
    results["Precision"].append(prec)
    results["Recall"].append(rec)

    print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    if acc > highest_accuracy:
        highest_accuracy = acc
        best_model_name = name
        best_y_pred = y_pred

# ==========================================
# PHASE C: Visualization
# ==========================================
print("\n--- PHASE C: Visualization ---")
sns.set_theme(style="whitegrid")

# 1. Model Comparison: Bar Chart
x = np.arange(len(model_names))
width = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, results["Accuracy"], width, label="Accuracy", color="#4C72B0")
ax.bar(x, results["Precision"], width, label="Precision", color="#55A868")
ax.bar(x + width, results["Recall"], width, label="Recall", color="#C44E52")
ax.set_ylabel("Scores")
ax.set_title("Model Comparison: Accuracy, Precision, and Recall")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend(loc="lower right")
plt.show()

# 2. Confusion Matrix Heatmap (for the Best Model)
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"],
)
plt.title(f"Confusion Matrix: {best_model_name}")
plt.show()

# 3. ROC Curve (Logistic Regression)
y_prob_logreg = models["Logistic Regression"].predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_logreg)
plt.figure(figsize=(7, 6))
plt.plot(
    fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc(fpr, tpr):.2f})"
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.show()
