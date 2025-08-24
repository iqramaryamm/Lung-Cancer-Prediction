import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import List, Optional, Any, Dict

sns.set(style="whitegrid")

class DataAnalyser:
    """Component for Exploratory Data Analysis"""

    def __init__(self, target_col: str = "Level"):
        self.target_col = target_col

    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        analysis_results = {}

        # Capture df.info() as string
        buffer = io.StringIO()
        df.info(buf=buffer)
        analysis_results['info'] = buffer.getvalue()

        # Missing values
        analysis_results['missing_values'] = df.isnull().sum()

        # Descriptive stats by target
        if self.target_col in df.columns:
            analysis_results['descriptive_stats'] = df.groupby(self.target_col).describe()
        else:
            analysis_results['descriptive_stats'] = None

        # Correlation matrix
        df_temp = df.copy()
        if df_temp[self.target_col].dtype == "object":
            df_temp[self.target_col] = df_temp[self.target_col].map({"low": 0, "medium": 1, "high": 2})
        analysis_results['correlation_matrix'] = df_temp.select_dtypes(include=[np.number]).corr()

        return analysis_results

    def describe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.describe(include="all").transpose()

    def class_balance(self, df: pd.DataFrame) -> pd.Series:
        return df[self.target_col].value_counts(normalize=True).rename("proportion")

    def print_analysis_summary(self, analysis_results: Dict[str, Any]):
        print(" Data Analysis Summary")

        print("\n Info:")
        print(analysis_results['info'])

        print("\n Missing Values in each column:")
        print(analysis_results['missing_values'])

        print("\n Descriptive Statistics by Level:")
        print(analysis_results['descriptive_stats'])

        print("\nCorrelation Matrix of all features:")
        print(analysis_results['correlation_matrix'])

    def correlation_heatmap(self, df: pd.DataFrame, figsize=(12, 10)):
        numeric_df = df.select_dtypes(include="number")
        corr = numeric_df.corr()

        # Create figure and axes first
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlGn",
            cbar_kws={"label": "Correlation Coefficient"}, ax=ax
        )
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        return fig   # ✅ return figure for Streamlit


    def hist_by_target(self, df: pd.DataFrame, cols: Optional[List[str]] = None, cols_per_row: int = 4):
        if cols is None:
            cols = [c for c in df.columns if c != self.target_col]

        n = len(cols)
        rows = (n + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.histplot(data=df, x=col, hue=self.target_col, multiple="stack", ax=axes[i])
            axes[i].set_title(f"{col} by {self.target_col}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        return fig   # ✅ return fig
