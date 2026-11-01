# ModelFoundry

ModelFoundry is a local AutoML workbench for tabular CSV data. Upload a dataset, choose the value to predict, select algorithms, train comparable pipelines, and download both the best fitted model and a Python script for reproducing the run.

## Features

- Classification and regression workflows with an editable task suggestion
- CSV preview, target selection, and row, column, missing value, and file size summaries
- Candidate models: Logistic Regression, Linear and Ridge Regression, Random Forest, Extra Trees, and Histogram Gradient Boosting
- Training-only imputing, scaling, and categorical encoding inside scikit-learn pipelines
- A shared held-out split and task-specific evaluation metrics
- Downloads for the selected run's generated Python script and best fitted `.joblib` model
- Runs locally; uploaded CSV data is not sent to a project backend

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a CSV, select a target, choose one or more algorithms, and click **Train and compare models**.

## Evaluation notes

Classification reports accuracy, balanced accuracy, and weighted F1. Regression reports MAE, RMSE, and R². Candidate models use the same random holdout split (seed 42); classification uses stratification when every class has enough examples. Models rank by weighted F1 or RMSE. This compact comparison is a starting point, not proof of generalization. Check for target leakage, class imbalance, time or group structure, and use validation that matches the intended setting.

The app removes rows with a missing target, imputes feature missingness within each training pipeline, standardizes numeric features, and one-hot encodes categorical features. Review the generated script and the data before using a downloaded model.

## Limits

This is a portfolio-scale tabular demo. It does not automatically detect leakage, choose a scientifically valid split for every dataset, tune hyperparameters, or guarantee future performance. Uploaded data and model files may contain sensitive information; keep them in an appropriately protected environment.
