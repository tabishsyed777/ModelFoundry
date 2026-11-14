# Development requirements

These requirements come from the project owner's instructions and apply to all work on this project.

- Follow test-driven development for behavior changes: write a test for the required behavior, run it and confirm the expected failure, implement the smallest change that passes, then refactor while keeping tests green.
- Always run unit/component tests when making product changes.
- Always run integration tests as well. Cover the real CLI process, file outputs, exit codes, and relevant failure paths rather than relying only on mocked calls.
- Before handing off a change, run both test groups and report their outcomes. Resolve failures caused by the change; identify any environment blocker explicitly.
- Keep drift reference data restricted to the training split. Do not change expected results merely to make failing implementation pass.

## Required checks

```sh
python -m unittest discover -s tests -p test_training.py -v
python -m unittest discover -s tests -p test_ops_unit.py -v
python -m unittest discover -s tests -p test_ops_integration.py -v
python -m unittest discover -s tests -p test_ops_cli.py -v
```

The demo is an additional check. Use a new output path for each run:

```sh
python -m modelops.demo --out runs/review
```

For serving/container changes, also build the image and run its integration test:

```sh
docker build -f Dockerfile.ops -t modelfoundry-ops:test .
MODELFOUNDRY_TEST_IMAGE=modelfoundry-ops:test python -m unittest discover -s tests -p test_ops_container.py -v
```
