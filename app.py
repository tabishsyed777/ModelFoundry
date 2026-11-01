from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from modelfoundry import CLASSIFIERS, REGRESSORS, export_model, generate_script, infer_task, train_models

st.set_page_config(page_title="ModelFoundry · AutoML Studio", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--ink:#e9eef8;--muted:#9caac0;--bg:#0b1018;--panel:#111a27;--line:#243248;--mint:#66e0c2;}
html,body,[class*="css"] {font-family:'Manrope',sans-serif}
.stApp {background:radial-gradient(ellipse at 10% 0%,#15313b 0%,#0b1018 36%,#0b1018 100%); color:var(--ink)}
.block-container {max-width:1420px;padding-top:2.1rem;padding-bottom:3rem}
.hero {padding:18px 0 20px}.eyebrow {font:500 12px 'DM Mono',monospace;letter-spacing:.2em;color:var(--mint);text-transform:uppercase}
.hero h1 {font-size:clamp(42px,6vw,70px);line-height:1.04;letter-spacing:-.055em;margin:12px 0;color:#f4f7fb}
.hero p {color:var(--muted);font-size:18px;max-width:800px;line-height:1.6;margin:0}
.card {background:linear-gradient(145deg,rgba(20,31,47,.97),rgba(14,22,34,.97));border:1px solid var(--line);border-radius:18px;padding:22px 24px;margin-bottom:16px;box-shadow:0 16px 45px rgba(0,0,0,.13)}
.tag {display:inline-block;color:var(--mint);border:1px solid #285d58;background:#11302d;border-radius:99px;padding:7px 12px;font:12px 'DM Mono',monospace}
.small {color:var(--muted);font-size:14px}.metric-label {font-size:13px;color:var(--muted)}
div[data-testid="stMetric"] {background:#101a27;border:1px solid #243248;border-radius:14px;padding:14px 16px}
div[data-testid="stMetricValue"] {color:#f4f7fb}
div.stButton>button[kind="primary"] {background:var(--mint);color:#07130f;border:0;border-radius:10px;font-weight:800;height:48px}
div.stButton>button {border-radius:10px}
section[data-testid="stSidebar"] {background:#0d1520;border-right:1px solid #202d3e}
code,pre {font-family:'DM Mono',monospace}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">ModelFoundry · machine learning workbench</div>
  <h1>From raw data<br/>to a trained model.</h1>
  <p>Upload a CSV, choose what to predict, compare algorithms, and export the exact Python workflow behind your results.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ ModelFoundry")
    st.caption("A local AutoML workspace")
    st.divider()
    st.markdown("**1 · Add data**")
    uploaded = st.file_uploader("Choose a CSV dataset", type=["csv"], help="CSV files are processed in this browser session and are not uploaded to a remote service.")
    st.markdown('<p class="small">Your dataset stays in this app session. Use a target column with at least two values.</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**2 · Configure training**")

if uploaded is None:
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="card"><span class="tag">READY FOR YOUR CSV</span><h3 style="margin-top:18px">A repeatable model comparison</h3><p class="small">ModelFoundry builds preprocessing and model pipelines together, so imputing and encoding happen inside each training split.</p><ol class="small"><li>Upload a table</li><li>Select the value you want to predict</li><li>Choose candidate algorithms</li><li>Train, compare, and export</li></ol></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="eyebrow">SUPPORTED TASKS</div><h3>Classification</h3><p class="small">Predict a category such as churn, species, or approval status.</p><h3>Regression</h3><p class="small">Predict a numeric value such as price, demand, or duration.</p><hr/><p class="small">No sample result is presented as a real benchmark. Upload a dataset to create your own run.</p></div>', unsafe_allow_html=True)
    st.stop()

try:
    data = pd.read_csv(uploaded, low_memory=False)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if len(data) < 12 or data.shape[1] < 2:
    st.error("The dataset needs at least 12 rows and two columns (features plus a target).")
    st.stop()

data.columns = [str(c).strip() for c in data.columns]
if len(set(data.columns)) != len(data.columns):
    st.error("Column names must be unique. Rename duplicate columns in your CSV, then upload it again.")
    st.stop()
data = data.replace([float("inf"), float("-inf")], pd.NA)

overview = st.columns(4)
overview[0].metric("Rows", f"{len(data):,}")
overview[1].metric("Columns", f"{data.shape[1]:,}")
overview[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
overview[3].metric("File size", f"{uploaded.size / 1024:.0f} KB")

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(data.head(12), use_container_width=True, hide_index=True)

with st.sidebar:
    target = st.selectbox("Target column", options=data.columns, index=len(data.columns)-1, help="The target is the value the models will learn to predict.")
    suggested = infer_task(data[target])
    task_option = st.selectbox("Prediction task", ["Auto-detect", "Classification", "Regression"], index=0)
    task = suggested if task_option == "Auto-detect" else task_option.lower()
    catalog = CLASSIFIERS if task == "classification" else REGRESSORS
    defaults = list(catalog)[:3]
    algorithms = st.multiselect("Algorithms to compare", list(catalog), default=defaults)
    test_percent = st.slider("Holdout data", min_value=15, max_value=35, value=20, step=5, help="This share is held aside for evaluation and is not used to fit the models.")
    st.caption(f"Suggested task: {suggested.title()}")
    train = st.button("Train and compare models", type="primary", use_container_width=True, disabled=not algorithms)

st.markdown(f'<span class="tag">{task.upper()}</span> &nbsp; <span class="small">Target: {target} · {len(algorithms)} algorithm(s) selected</span>', unsafe_allow_html=True)
if train:
    try:
        with st.spinner("Preparing a shared holdout split and fitting the selected models…"):
            metrics, result = train_models(data, target, task, algorithms, test_percent / 100)
        st.session_state["run"] = {"metrics": metrics, "result": result, "data": data, "algorithms": algorithms, "test_fraction": test_percent/100}
    except Exception as exc:
        st.error(str(exc))

run = st.session_state.get("run")
if run:
    metrics = run["metrics"]
    result = run["result"]
    best_name = str(metrics.iloc[0]["Algorithm"])
    best_score = metrics.iloc[0][result["primary_metric"]]
    st.markdown("## Training results")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Top model", best_name)
    c2.metric(result["primary_metric"], f"{best_score:.4f}", help="Models are ranked by weighted F1 for classification or RMSE for regression.")
    c3.metric("Training rows", f"{result['train_rows']:,}")
    c4.metric("Held-out rows", f"{result['test_rows']:,}")
    st.dataframe(metrics.style.format({c:"{:.4f}" for c in metrics.columns if c != "Algorithm"}), use_container_width=True, hide_index=True)
    if result["removed_target_rows"]:
        st.caption(f"{result['removed_target_rows']:,} row(s) with an empty target were excluded before splitting.")
    st.caption("These holdout metrics are a starting point, not a guarantee of performance on future or external data. Review class balance, leakage, and the split design before relying on a model.")

    st.markdown("## Export this run")
    st.write("Download the fitted top model or inspect the generated Python script to reproduce the preprocessing, split, algorithms, and comparison.")
    script = generate_script(target, task, run["algorithms"], run["test_fraction"])
    d1,d2 = st.columns(2)
    d1.download_button("Download Python training script", script, file_name="modelfoundry_run.py", mime="text/x-python", use_container_width=True)
    d2.download_button("Download top model (.joblib)", export_model(result["models"][best_name]), file_name="best_model.joblib", mime="application/octet-stream", use_container_width=True)
    with st.expander("View generated Python code"):
        st.code(script, language="python", line_numbers=True)

st.markdown("<div style='margin-top:48px;border-top:1px solid #243248;padding-top:18px;color:#8796ad;font-size:13px'>ModelFoundry trains locally in the app process. Treat uploaded datasets and exported models as sensitive files. Validate the workflow on data suited to your use case.</div>", unsafe_allow_html=True)
