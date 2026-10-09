# ============================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sns.set_theme(style="whitegrid")



iris = load_iris()
X = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
y = pd.Series(iris.target, name="species")

print("Features shape:", X.shape)
print("Target classes:", list(iris.target_names))



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)} | Testing samples: {len(X_test)}")



dt = DecisionTreeClassifier(
    criterion="gini",       
    max_depth=3,           
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
)

dt.fit(X_train, y_train)



y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)



acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro")
rec_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

print("\n Evaluation Metrics")
print(f"Accuracy (overall):        {acc:.4f}")
print(f"Precision (macro avg):     {prec_macro:.4f}")
print(f"Recall (macro avg):        {rec_macro:.4f}")
print(f"F1-score (macro avg):      {f1_macro:.4f}")


print("\n--- Per-Class Precision / Recall / F1 ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names, digits=4))


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names))





