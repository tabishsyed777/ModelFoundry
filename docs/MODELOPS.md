# ModelFoundry deployment and monitoring

This extension serves pipelines trained by the existing ModelFoundry trainer. It adds immutable model versions, an HTTP prediction API, and persistent per-version latency and input-distribution summaries. Training and inference run locally without an OpenAI key. The Streamlit trainer remains available; this release adds a separate CLI/API workflow.

## Setup and synthetic demo

Requires Python 3.11 or 3.12. Create and activate a virtual environment, then:

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p test_training.py -v
python -m unittest discover -s tests -p test_ops_unit.py -v
python -m unittest discover -s tests -p test_ops_integration.py -v
python -m unittest discover -s tests -p test_ops_cli.py -v
python -m modelops.demo
```

Open `runs/modelops-demo/report.html`. The demo trains two models on the existing synthetic churn dataset, registers two versions, compares API predictions with direct pipeline predictions, submits shifted inputs, and verifies metrics after recreating the API application. Output paths must be new; use `--out runs/another-demo` to repeat. No model service or OpenAI API is called. The demo uses FastAPI's in-process HTTP client; a separate integration test starts a real Uvicorn process.

## Train and register your CSV

```sh
python -m modelops.cli \
  --csv your_dataset.csv \
  --target churned \
  --task classification \
  --algorithm 'Logistic Regression' \
  --registry runs/registry
```

Use a supported algorithm from the original trainer; regression supports `--task regression --algorithm 'Ridge Regression'`, for example. Each successful registration prints a new immutable version ID and its manifest, including creation time, holdout metrics, feature types, artifact checksum, and runtime versions. The drift reference uses only the training split, excluding holdout data and the target. Registry operations are local CLI calls; the HTTP API has no artifact-upload or registration endpoint.

## Serve and predict

```sh
export MODELFOUNDRY_REGISTRY=runs/registry
# Optional for a local session; required before exposing the service beyond your machine:
export MODELFOUNDRY_API_KEY='replace-with-your-own-token'
python -m uvicorn modelops.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API schema. When a key is configured, use its Authorize control. Requests to protected endpoints require `Authorization: Bearer <your-token>`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Public process liveness, without model loading |
| `GET /models` | List registered versions and metadata |
| `GET /models/{version}` | Describe an explicit version |
| `POST /predict/{version}` | Predict a batch using exactly that version |
| `GET /monitor/{version}` | Persistent version-scoped request, latency, and input-drift summaries |

For the synthetic churn model, the prediction body is:

```json
{"rows":[{"tenure_months":24,"monthly_charge":75.5,"support_calls_90d":2,"late_payments_12m":1,"plan":"Plus"}]}
```

Every row must contain exactly the registered feature names. Numeric values must be finite JSON numbers, boolean features require booleans, and categorical features require strings. Nulls are allowed for the pipeline's fitted imputation. Unknown categorical values use the existing encoder's `handle_unknown='ignore'` behavior. Batches contain 1–1000 rows. Invalid inputs return 422, unknown versions return 404, and missing/corrupt/incompatible artifacts return 503. Rejected requests do not increment successful-request metrics.

Selecting a previous version ID serves that model again; no implicit “latest” alias or automatic promotion is implemented.

## Drift and latency semantics

Numeric profiles use training quintile thresholds, with duplicate thresholds removed, plus a missing-value bucket. Categorical profiles retain up to 30 frequent labels plus distinct other/missing buckets. Boolean profiles preserve true/false semantics. Drift is total variation distance: half the sum of absolute differences between current and reference bucket proportions. A feature is flagged when this distance is greater than 0.2. This fixed heuristic is not a significance test, a calibrated alert, or proof of model degradation; small batches can trigger it.

Prediction responses include per-batch drift. The monitor accumulates feature histograms across **all successful requests** for that version. It is not a recent-time drift window; old observations can dilute new changes. Histogram aggregation loses relationships between columns and can miss shifts within numeric bins or within the categorical “other” bucket. Ground-truth labels are not collected, so predictive accuracy is not monitored.

Latency measures validation, artifact loading/checksum verification, and inference inside the request handler. It excludes network time, response serialization, and the metrics database write. Mean latency covers all successful requests; p50/p95 use the most recent 100 successful request durations. Counts and histograms persist in SQLite, using a transaction for each aggregate update. Raw prediction input rows and predictions are not stored in monitoring tables.

## Container deployment

Build and run a self-contained image:

```sh
docker build -f Dockerfile.ops -t modelfoundry-ops .
docker volume create modelfoundry-data
```

Train/register within the same image used for serving. For example, mount a trusted input folder containing your CSV:

```sh
docker run --rm \
  -v "$PWD/input:/input:ro" -v modelfoundry-data:/data \
  modelfoundry-ops python -m modelops.cli \
  --csv /input/your_dataset.csv --target churned --task classification \
  --algorithm 'Logistic Regression' --registry /data/registry

docker run --rm -p 127.0.0.1:8000:8000 \
  -v modelfoundry-data:/data \
  -e MODELFOUNDRY_API_KEY \
  modelfoundry-ops
```

On networks with a custom certificate authority, pass a trusted CA bundle as a temporary build secret: `docker build --secret id=pip_ca,src=/path/to/trusted-ca.pem -f Dockerfile.ops -t modelfoundry-ops .`. TLS verification stays enabled and the bundle is not stored in the image.

The image runs as a non-root user. Keep the named volume to retain versions and metrics across restarts. The API checks exact Python, scikit-learn, NumPy, pandas, and joblib versions before loading an artifact. Moving artifacts from a different runtime is rejected; retrain using the serving image. Dependencies use bounded version ranges, and the base image tag can change: preserve the built image digest for a deployment. This repository does not claim a fully pinned supply chain.

Verify the real container training and serving path:

```sh
MODELFOUNDRY_TEST_IMAGE=modelfoundry-ops python -m unittest discover -s tests -p test_ops_container.py -v
```

That test needs a local Docker daemon and a Linux host with `os.getuid`; it trains inside the image and calls its authenticated HTTP service over a loopback port. CI builds the image and runs this test as a separate required job.

## Test evidence

See [validation record](MODELOPS-VALIDATION.md) for the tests-first sequence, test groups, and reproducible commands.

## Operational limits

Joblib artifacts can execute code when loaded. Only operator-owned, trusted artifacts belong in this registry; checksums detect accidental changes and do not authenticate a maliciously replaced registry. Protect the registry directory and backups. Profiles can contain feature names and common category labels; model files and metadata may also carry sensitive information. Monitoring does not record raw requests, but registry artifacts are not anonymous.

This is a single-host portfolio deployment. Public hosting still needs TLS, a reverse proxy with body/time/rate limits, credential management, restricted filesystem access, backups, and operational ownership. No public/cloud deployment, autoscaling, alert delivery, ground-truth performance monitor, access audit log, or automated retraining is configured. Authentication is optional for loopback use and configured through a bearer token when needed; there is no multi-user authorization system.
