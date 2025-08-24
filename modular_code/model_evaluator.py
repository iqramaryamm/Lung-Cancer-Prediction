# model_evaluator.py
from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

sns.set(style="whitegrid")

def evaluate_models(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Returns a dataframe of metrics for each model and plots a confusion matrix per model.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1_macro": f1_score(y_test, y_pred, average="macro"),
            "Recall_macro": recall_score(y_test, y_pred, average="macro"),
            "Precision_macro": precision_score(y_test, y_pred, average="macro"),
        })

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        # Optional: print detailed classification report
        print(f"\n=== Classification Report: {name} ===")
        print(classification_report(y_test, y_pred, digits=3))

    results_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    # Bar chart for quick compare
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=results_df, x="Model", y="Accuracy")
    ax.set_title("Comparsion of Model Accuracy")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    return results_df
