# Deployment and monitoring validation

The implementation followed tests-first development. Before the serving modules existed, the new unit/integration suite failed on missing `modelops` imports and on the missing `X_train` training-reference result. The demo CLI test failed because its entry point did not exist. The implementation was then added to satisfy those tests.

Additional regression tests reproduced an oversized-integer validation error and a non-finite prediction being counted before its response failed. The fixes reject both as input/prediction validation errors and preserve the successful-request counters. The container test first failed on a missing image and later on non-root source-file permissions; the image was completed and its permissions corrected before that test passed.

## Checks

| Group | Coverage |
| --- | --- |
| Training reference | Uses exactly the training split and excludes holdout rows/target |
| Profile unit tests | Identical/shifted inputs, missing/unknown categories, strict numeric/boolean/string schema |
| Registry unit tests | Unique immutable IDs, artifact checksums, loading after restart, version-scoped persisted metrics |
| HTTP integration | Real fitted classification/regression predictions, schema errors, authentication, corrupt-artifact rejection, non-finite prediction rejection, drift, persistence |
| Live server integration | Starts a Uvicorn subprocess and sends prediction/monitor requests over loopback HTTP |
| CLI integration | CSV training/registration and the complete synthetic demo, including refusal to overwrite existing evidence |
| Container integration | Trains in the image, serves with authentication, predicts over a published loopback port, and checks persisted request/row counts |

The local full suite contains 15 tests: seven unit/component tests and eight integration tests including the opt-in real container test. Run the entire suite with:

```sh
docker build -f Dockerfile.ops -t modelfoundry-ops:test .
MODELFOUNDRY_TEST_IMAGE=modelfoundry-ops:test python -m unittest discover -s tests -v
```

Without the image environment variable, the container test is explicitly skipped; the other 14 tests run. CI runs unit/component and HTTP/CLI integration checks on Python 3.11 and 3.12 and builds/tests the real image in a separate job.

The synthetic demo registers two model versions, matches served predictions to direct model predictions, detects an intentionally large input shift, and checks metrics after an application restart. Its report was opened in Chromium. These are reproducibility and behavior checks, not evidence of production readiness or general model quality. Latency measurements vary by machine and include artifact loading; no cloud deployment or performance benchmark was performed.
