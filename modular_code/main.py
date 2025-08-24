from data_loader import DataLoader
from data_analyser import DataAnalyser
from model_trainer import train_models
from model_evaluator import evaluate_models

# 1) Load & prep

loader = DataLoader(target_col="Level", id_col="Patient Id")
df, X_train, X_test, y_train, y_test = loader.load_and_prepare("cancer_patient_data_sets.xlsx")

# 2) (Optional) EDA
analyser = DataAnalyser(target_col="Level")

print("Data Analysis",analyser.analyze_data(df))
print("Description",analyser.describe(df).head())
print("Class balance:\n", analyser.class_balance(df))

# analyser.correlation_heatmap(df)
# analyser.hist_by_target(df)
# analyser.pairplot_numeric(df, max_vars=8)

# 3) Train
models = train_models(X_train, y_train, use_grid_search=False, cv=3)

# 4) Evaluate
results_df = evaluate_models(models, X_test, y_test)
print(results_df)
