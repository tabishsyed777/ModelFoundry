"""Synthetic train → register → HTTP API → monitor reproducibility demo."""
import argparse
import html
import json
from pathlib import Path
from fastapi.testclient import TestClient
from modelfoundry import sample_dataset,train_models
from .api import create_app
from .registry import Registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',default='runs/modelops-demo')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error('Output directory already exists; choose a new --out path')
    out.mkdir(parents=True)
    data = sample_dataset()
    registry = Registry(out/'registry')
    manifests = []
    for algorithm in ['Logistic Regression','Random Forest']:
        metrics,result = train_models(data,'churned','classification',[algorithm],.25)
        manifest = registry.register(result['models'][algorithm],result['X_train'],
                                     {'algorithm':algorithm,'task':'classification','target':'churned','holdout_metrics':metrics.iloc[0].to_dict()})
        manifests.append(manifest)
    version = manifests[-1]['version']
    rows = result['X_test'].head(8).to_dict(orient='records')
    with TestClient(create_app(out/'registry',api_key='')) as client:
        response = client.post(f'/predict/{version}',json={'rows':rows})
        response.raise_for_status()
        predictions_match = response.json()['predictions']==result['models']['Random Forest'].predict(result['X_test'].head(8)).tolist()
        shifted = [dict(row,monthly_charge=10000,plan='UNSEEN') for row in rows]
        response = client.post(f'/predict/{version}',json={'rows':shifted})
        response.raise_for_status()
    with TestClient(create_app(out/'registry',api_key='')) as restarted:
        monitor = restarted.get(f'/monitor/{version}').json()
    evidence = {'versions_registered':len(registry.list_versions()),'predictions_match':predictions_match,
                'drift_detected':monitor['features']['monthly_charge']['drifted'] and monitor['features']['plan']['drifted'],
                'metrics_survive_restart':monitor['requests']==2 and monitor['rows']==16,
                'model_calls':0,'models':manifests,'monitor':monitor,
                'scope':'Synthetic demonstration. Drift compares input distributions, not model accuracy.'}
    text = json.dumps(evidence,indent=2,allow_nan=False)
    (out/'evidence.json').write_text(text+'\n',encoding='utf-8')
    (out/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ModelFoundry deployment evidence</title><style>body{{font:16px system-ui;background:#0c1728;color:#e5efff;max-width:1000px;margin:48px auto;padding:24px}}h1{{color:#83ead1}}pre{{background:#18283d;white-space:pre-wrap;overflow-wrap:anywhere;padding:24px;border-radius:12px}}a{{color:#83ead1}}</style><h1>ModelFoundry / Deployment &amp; Monitoring</h1><p>Two versioned models · real trained predictions · persisted latency and input histograms</p><p>Prediction match: {predictions_match} · Input shift detected: {evidence['drift_detected']} · Metrics survived restart: {evidence['metrics_survive_restart']}</p><p>Input drift is not a measurement of predictive accuracy.</p><a href="evidence.json">JSON evidence</a><pre>{html.escape(text)}</pre></html>''',encoding='utf-8')
    print(f'Evidence: {out / "report.html"}')
    return 0 if all(evidence[key] for key in ['predictions_match','drift_detected','metrics_survive_restart']) else 1


if __name__=='__main__':
    raise SystemExit(main())
