# Lung Cancer Prediction 

This project provides a modular Python pipeline for loading, preprocessing, analyzing, training, and evaluating machine learning models on a cancer patient dataset. The pipeline is designed to be reusable, prevent data leakage, and support exploratory data analysis (EDA) and model evaluation with visualizations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Components](#components)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Dependencies](#dependencies)
- [Assumptions and Notes](#assumptions-and-notes)

## Overview
The pipeline processes a cancer patient dataset (in `.csv` or `.xlsx` format) with a target column (`Level`) indicating cancer severity (e.g., "low", "medium", "high"). It includes four main components:
- **DataLoader**: Loads and preprocesses the dataset.
- **DataAnalyser**: Performs EDA with summary statistics and visualizations.
- **ModelTrainer**: Trains multiple machine learning models with optional hyperparameter tuning.
- **ModelEvaluator**: Evaluates models using metrics and visualizations.

## Features
- Supports `.csv` and `.xlsx` file formats.
- Encodes target variable (`Level`) as numerical values (`low: 0, medium: 1, high: 2`).
- Handles missing values and removes Patient-ID like columns to prevent leakage.
- Provides EDA with correlation heatmaps, histograms, and class balance analysis.
- Trains multiple models (Logistic Regression, Random Forest, Decision Tree, KNN) with pipelines to ensure proper scaling.
- Evaluates models with accuracy, F1-score, precision, recall, confusion matrices, and classification reports.
- Saves trained models to `.pkl` files.

## Components
1. **DataLoader** (`data_loader.py`):
   - Loads dataset and preprocesses it by encoding the target, dropping irrelevant columns, and handling missing values.
   - Splits data into training and test sets with optional stratification.

2. **DataAnalyser** (`data_analyser.py`):
   - Generates dataset summaries, missing value counts, and correlation matrices.
   - Visualizes data with correlation heatmaps and histograms by target.

3. **ModelTrainer** (`model_trainer.py`):
   - Trains a suite of models with `StandardScaler` in pipelines to avoid leakage.
   - Supports grid search for hyperparameter tuning.
   - Checks for and removes leakage-prone columns.

4. **ModelEvaluator** (`model_evaluator.py`):
   - Computes performance metrics (accuracy, F1-score, precision, recall).
   - Generates confusion matrices and a bar chart comparing model accuracies.
   
## Usage
The pipeline can be used in a Python script or Jupyter notebook. Below is a basic example of how i have used the components together.

### Example
```python
from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import train_models
from model_evaluator import evaluate_models
import matplotlib.pyplot as plt

# Load and preprocess data
loader = DataLoader()
df, X_train, X_test, y_train, y_test = loader.load_and_prepare("cancer_data.csv")

# Perform EDA
analyser = DataAnalyser()
results = analyser.analyze_data(df)
analyser.print_analysis_summary(results)
fig = analyser.correlation_heatmap(df)
plt.show()
fig = analyser.hist_by_target(df)
plt.show()

# Train models
models = train_models(X_train, y_train, use_grid_search=True)

# Evaluate models
results_df = evaluate_models(models, X_test, y_test)
print(results_df)
```

This script loads a dataset, performs EDA, trains models with hyperparameter tuning, and evaluates their performance.

## Dependencies
- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical operations.
- `matplotlib`: Plotting and visualization.
- `seaborn`: Enhanced visualizations.
- `scikit-learn`: Machine learning models and metrics.
- `openpyxl`: Support for `.xlsx` files.

```

## Assumptions and Notes
- The dataset has  a `Level` column with values `low`, `medium`, or `high` (case-insensitive).
- `Patient Id` column is dropped during preprocessing.
- Missing values in numeric features are imputed with medians, excluding the target column.
- Models use default hyperparameters unless `use_grid_search=True` is specified.
- Visualizations require a matplotlib backend (e.g., Jupyter or Streamlit).
- Grid search is computationally intensive; use `n_jobs=-1` for parallel processing.
- The pipeline assumes a clean dataset; additional preprocessing may be needed for specific cases.


Why I Consider Logistic Regression?

Logistic Regression is a foundational linear model often used for binary classification  Here's why it's a strong choice for this problem:

Interpretability for Insights: It provides coefficients (odds ratios) for each feature, directly showing how variables like smoking history or genetic markers increase/decrease cancer risk. This helps answer "what causes cancer" by quantifying feature impacts (eg high coefficient for age might indicate it's a key risk factor).

Simplicity and Efficiency: It assumes a linear relationship between features and the log-odds of the outcome, making it fast to train on moderate-sized datasets. No need for heavy computation, which is ideal for quick iterations in research.

Handling Imbalanced Data: Cancer datasets often have class imbalance (e.g., more non-cancer cases). Logistic Regression works well with techniques like weighting or thresholding to handle this.


Why Consider Decision Trees?

Handling Non-Linearity and Mixed Data: Unlike linear models, trees capture complex, non-linear relationships without assumptions. Cancer data often includes categorical (eg: gender) and continuous (e.g., biomarker levels) features, which trees handle natively.

Robustness to Outliers/Missing Data: Common in patient datasets; Decision trees can manage this without much preprocessing.



Why Consider K-Nearest Neighbors (KNN)?

Instance-based learner that classifies new data based on similarity to 'k' nearest training points. It's useful here for:

Exploratory Analysis: Helps visualize data clusters (e.g., via distance metrics), revealing subgroups prone to certain cancers or treatments.

Ease of Implementation: Simple to tune (mainly 'k' and distance metric), and effective for small-to-medium datasets where computational cost isn't prohibitive.


Why not other Approaches ?

Lack of Interpretability: Models like Neural Networks are "black boxes," making it hard to extract clear insights into causes/treatments (e.g., why a feature matters).

Overfitting and Data Requirements: Cancer datasets are often small/high-dimensional (curse of dimensionality). Complex models like Deep Learning need massive data to avoid overfitting and generalize; KNN/Decision Trees perform better on limited samples without heavy regularization

Scalability to High Dimensions: Cancer data can have many features,  statistical methods struggle with multicollinearity, while Logistic Regression/KNN manage via  feature selection.