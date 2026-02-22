# =========================================================
# CREDIT CARD FRAUD DETECTION PROJECT
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("creditcard.csv")
print("Dataset Loaded Successfully!")

print("\nClass Distribution:")
print(df['Class'].value_counts())

# =========================================================
# 2. SPLIT FEATURES & TARGET
# =========================================================

X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Amount column (important)
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# =========================================================
# 3. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# =========================================================
# 4. HANDLE IMBALANCE USING SMOTE
# =========================================================

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())

# =========================================================
# 5. LOGISTIC REGRESSION
# =========================================================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_res, y_train_res)

y_pred_lr = lr_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# =========================================================
# 6. DECISION TREE
# =========================================================

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train_res, y_train_res)

y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# =========================================================
# 7. RANDOM FOREST (BEST MODEL)
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train_res, y_train_res)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))

# =========================================================
# 8. AUC SCORE
# =========================================================

rf_auc = roc_auc_score(y_test, y_prob_rf)
print("AUC Score:", rf_auc)

# =========================================================
# 9. ROC CURVE
# =========================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.show()

# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

importances = rf_model.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feat_df.head(10))

plt.figure()
plt.bar(feat_df['Feature'][:10], feat_df['Importance'][:10])
plt.xticks(rotation=90)
plt.title("Top 10 Important Features")
plt.show()

# =========================================================
# 11. MODEL COMPARISON TABLE
# =========================================================

results = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf)
    ]
}

results_df = pd.DataFrame(results)

print("\nModel Comparison:\n")
print(results_df)

print("\nProject Completed Successfully 🚀")
