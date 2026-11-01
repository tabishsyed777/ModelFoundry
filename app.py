from __future__ import annotations

import pandas as pd
import streamlit as st

from agent import explain_run, openai_settings, prepare_run
from modelfoundry import CLASSIFIERS, REGRESSORS, export_model, generate_script, sample_dataset, train_models

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
  <p>Upload a CSV and choose an algorithm. An OpenAI agent plans the workflow, trains a trusted local pipeline, and explains the generated Python.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ ModelFoundry")
    st.caption("A local AutoML workspace")
    st.divider()
    st.markdown("**1 · Add data**")
    uploaded = st.file_uploader("Choose a CSV dataset", type=["csv"], help="Raw data rows stay in the local app. If the OpenAI agent is enabled, only column names and aggregate schema details are sent to OpenAI.")
    if st.button("Try a sample dataset", use_container_width=True):
        st.session_state["sample_mode"] = True
        st.rerun()
    api_key, openai_model = openai_settings()
    if api_key:
        st.markdown(f'<p class="small">OpenAI agent connected · {openai_model}</p>', unsafe_allow_html=True)
        use_agent = st.checkbox(
            "Send schema and run results to OpenAI for agent guidance",
            value=False,
            help="This sends column names, data types, missing percentages, distinct counts, the selected model, holdout metrics, and generated training code. Dataset rows and cell values stay local. API usage may incur charges.",
        )
    else:
        st.markdown('<p class="small">Local trainer ready · add an OpenAI API key to enable the agent.</p>', unsafe_allow_html=True)
        use_agent = False
    st.divider()
    st.markdown("**2 · Configure training**")

if uploaded is None and not st.session_state.get("sample_mode", False):
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="card"><span class="tag">READY FOR YOUR CSV</span><h3 style="margin-top:18px">An agent-led training run</h3><p class="small">Choose a model and let the agent coordinate a reproducible local pipeline. You can review or change its target-column suggestion in Advanced options.</p><ol class="small"><li>Upload a table</li><li>Choose a classification or regression algorithm</li><li>Let the agent prepare and run the workflow</li><li>Review metrics, syntax notes, and generated code</li></ol></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="eyebrow">SUPPORTED TASKS</div><h3>Classification</h3><p class="small">Predict a category such as churn, species, or approval status.</p><h3>Regression</h3><p class="small">Predict a numeric value such as price, demand, or duration.</p><hr/><p class="small">No sample result is presented as a real benchmark. Upload a dataset to create your own run.</p></div>', unsafe_allow_html=True)
    st.stop()

if uploaded is not None:
    st.session_state["sample_mode"] = False
    try:
        data = pd.read_csv(uploaded, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()
else:
    data = sample_dataset()
    st.info("Synthetic example loaded for demonstration. Upload your own CSV any time to replace it.")

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
overview[3].metric("Dataset source", "Synthetic" if uploaded is None else f"{uploaded.size / 1024:.0f} KB")

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(data.head(12), use_container_width=True, hide_index=True)

with st.sidebar:
    all_algorithms = [("classification", name) for name in CLASSIFIERS] + [("regression", name) for name in REGRESSORS]
    selected_task, selected_algorithm = st.selectbox(
        "Choose an algorithm",
        all_algorithms,
        index=1,
        format_func=lambda item: f"{item[1]} · {item[0].title()}",
        help="The model family determines whether this is a classification or regression run.",
    )
    suggested_target = str(data.columns[-1])
    with st.expander("Advanced options", expanded=False):
        choose_target = st.checkbox("Choose the target column myself", value=False)
        if choose_target:
            target_override = st.selectbox("Target column", options=list(data.columns), index=len(data.columns)-1)
        else:
            target_override = None
            st.caption(f"The agent will suggest a target; local fallback: {suggested_target}")
        test_percent = st.slider("Holdout data", min_value=15, max_value=35, value=20, step=5, help="This share is held aside for evaluation and is not used to fit the model.")
    api_key, openai_model = openai_settings()
    button_label = "Plan and train with AI agent" if api_key and use_agent else "Train selected model locally"
    train = st.button(button_label, type="primary", use_container_width=True)

st.markdown(f'<span class="tag">{selected_task.upper()}</span> &nbsp; <span class="small">Model: {selected_algorithm} · target suggestion: {target_override or suggested_target}</span>', unsafe_allow_html=True)
if api_key and use_agent:
    st.caption("For this run, OpenAI receives column names, schema summaries, the selected model, holdout metrics, and generated code. Raw dataset rows are not sent. API usage may incur charges.")
else:
    with st.expander("Connect the OpenAI agent"):
        st.markdown("Add `OPENAI_API_KEY` to a local `.env` file using `.env.example` as a guide, then restart the app. The local trainer works without an API key.")
if train:
    try:
        agent_plan = None
        client = None
        agent_response_call = None
        if api_key and use_agent:
            with st.spinner("The agent is reviewing the dataset schema and assembling the training workflow…"):
                client, model_name, agent_response_call, agent_plan = prepare_run(
                    data, selected_algorithm, selected_task, target_override or suggested_target
                )
            target = target_override or agent_plan["target_column"]
        else:
            model_name = None
            target = target_override or suggested_target
        with st.spinner("Fitting the selected algorithm on a shared holdout split…"):
            metrics, result = train_models(data, target, selected_task, [selected_algorithm], test_percent / 100)
        script = generate_script(target, selected_task, [selected_algorithm], test_percent / 100)
        report = ""
        if api_key and use_agent and agent_plan:
            with st.spinner("The agent is explaining the model syntax and holdout results…"):
                report = explain_run(
                    client, model_name, agent_response_call, agent_plan,
                    {"metrics": metrics.to_dict(orient="records"), "train_rows": result["train_rows"], "test_rows": result["test_rows"], "target": target, "algorithm": selected_algorithm, "task": selected_task, "generated_code": script},
                )
        st.session_state["run"] = {"metrics": metrics, "result": result, "algorithms": [selected_algorithm], "test_fraction": test_percent/100, "target": target, "task": selected_task, "agent_plan": agent_plan, "agent_report": report, "script": script}
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
    c1.metric("Trained model", best_name)
    c2.metric(result["primary_metric"], f"{best_score:.4f}", help="Classification uses weighted F1 as the primary metric; regression uses RMSE.")
    c3.metric("Training rows", f"{result['train_rows']:,}")
    c4.metric("Held-out rows", f"{result['test_rows']:,}")
    st.dataframe(metrics.style.format({c:"{:.4f}" for c in metrics.columns if c != "Algorithm"}), use_container_width=True, hide_index=True)
    if result["removed_target_rows"]:
        st.caption(f"{result['removed_target_rows']:,} row(s) with an empty target were excluded before splitting.")
    if run.get("agent_plan"):
        st.markdown(f"**Agent target suggestion:** `{run['target']}` — {run['agent_plan']['target_reason']}")
        st.markdown("**Workflow syntax breakdown**")
        for step in run["agent_plan"]["workflow_steps"]:
            st.markdown(f"- {step}")
        if run.get("agent_report"):
            st.markdown("### Agent explanation")
            st.markdown(run["agent_report"])
    st.caption("These holdout metrics are a starting point, not a guarantee of performance on future or external data. Review class balance, leakage, and the split design before relying on a model.")

    st.markdown("## Export this run")
    st.write("Download the fitted model or inspect the generated Python script to reproduce this training run.")
    script = run["script"]
    d1,d2 = st.columns(2)
    d1.download_button("Download Python training script", script, file_name="modelfoundry_run.py", mime="text/x-python", use_container_width=True)
    d2.download_button("Download top model (.joblib)", export_model(result["models"][best_name]), file_name="best_model.joblib", mime="application/octet-stream", use_container_width=True)
    with st.expander("View generated Python code"):
        st.code(script, language="python", line_numbers=True)

st.markdown("<div style='margin-top:48px;border-top:1px solid #243248;padding-top:18px;color:#8796ad;font-size:13px'>ModelFoundry runs training locally. When enabled, its OpenAI agent receives schema metadata, not raw rows. Treat uploaded datasets and exported models as sensitive files. Validate workflows on data suited to your use case.</div>", unsafe_allow_html=True)
