# ModelFoundry

ModelFoundry is an AI-assisted workbench for tabular CSV data. Upload a dataset and choose a classification or regression algorithm. An OpenAI agent inspects the dataset schema, suggests a target, invokes the local training workflow, and explains the selected algorithm's syntax and evaluation results. Download the fitted model and generated Python script to reproduce the run.

## Features

- OpenAI agent uses function calling to prepare a constrained training run
- Agent suggests the prediction target from the dataset schema; users can override it
- Classification and regression algorithms: Logistic Regression, Linear and Ridge Regression, Random Forest, Extra Trees, and Histogram Gradient Boosting
- CSV preview, target suggestion, and row, column, missing value, and file size summaries
- Training-only imputing, scaling, and categorical encoding inside scikit-learn pipelines
- A shared held-out split and task-specific evaluation metrics for the selected algorithm
- Agent generated syntax walkthrough and holdout result explanation
- Downloads for the selected run's generated Python script and best fitted `.joblib` model
- Training runs locally; the OpenAI agent receives column names and aggregate schema metadata, not raw dataset rows

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a CSV, select a target, choose one or more algorithms, and click **Train and compare models**.

### Enable the OpenAI agent

Copy `.env.example` to `.env` and set your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key
MODELFOUNDRY_OPENAI_MODEL=gpt-6-luna
```

Restart Streamlit. The agent uses OpenAI's Responses API with a restricted function tool; training and generated Python execution stay in the local application. API requests can incur charges. The agent receives dataset column names, data types, missing-value percentages, distinct-value counts, the selected algorithm, and holdout metrics. It does not receive row values.

Without an API key, the selected model can still be trained locally, but AI target selection and explanations are unavailable.

## Evaluation notes

Classification reports accuracy, balanced accuracy, and weighted F1. Regression reports MAE, RMSE, and R². Training uses a random holdout split (seed 42); classification uses stratification when every class has enough examples. The metrics are a starting point, not proof of generalization. Check for target leakage, class imbalance, time or group structure, and use validation that matches the intended setting.

The app removes rows with a missing target, imputes feature missingness within each training pipeline, standardizes numeric features, and one-hot encodes categorical features. Review the generated script and the data before using a downloaded model.

## Limits

This is a portfolio-scale tabular demo. It does not automatically detect leakage, choose a scientifically valid split for every dataset, tune hyperparameters, or guarantee future performance. Review the AI's target suggestion and explanation. Uploaded data and model files may contain sensitive information; keep them in an appropriately protected environment.
