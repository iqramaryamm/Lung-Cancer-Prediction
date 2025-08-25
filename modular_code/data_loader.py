# data_loader.py
from __future__ import annotations
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    
    Loads and minimally preprocesses the cancer patient dataset.
    - Encodes Level -> {low:0, Medium:1, High:2}
    - Drops 'Patient Id' if present
    - Ensures target column is excluded from preprocessing of features (avoid leakage)
    - Returns X, y and the processed DataFrame for optional EDA
    """

    def __init__(self, target_col: str = "Level", id_col: str = "Patient Id"):
        self.target_col = target_col
        self.id_col = id_col

    def load(self, path) -> pd.DataFrame:
        if hasattr(path, "name"):
            filename = path.name
        else:
            filename = path

        if filename.endswith(".xlsx"):
            df = pd.read_excel(path)
        elif filename.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            raise ValueError("Unsupported file type. Use .xlsx or .csv")
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop ID column if present
        if self.id_col in df.columns:
            df = df.drop(columns=[self.id_col])

        # Normalize target label capitalization & map
        if self.target_col in df.columns:
            df[self.target_col] = (
                df[self.target_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"low": 0, "medium": 1, "high": 2})
            )
        else:
            raise KeyError(f"Target column '{self.target_col}' not found.")

        # Drop rows with NA in target
        df = df.dropna(subset=[self.target_col])

        # Fill numeric NA with medians (EXCLUDING target to avoid leakage!)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if self.target_col in num_cols:
            num_cols.remove(self.target_col)   # ✅ prevents leakage
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        return df

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None,
        )

    def load_and_prepare(
        self, path: str, test_size: float = 0.3, random_state: int = 42
    ):
        df = self.load(path)
        df = self.preprocess(df)
        X_train, X_test, y_train, y_test = self.split(
            df, test_size=test_size, random_state=random_state
        )
        return df, X_train, X_test, y_train, y_test
