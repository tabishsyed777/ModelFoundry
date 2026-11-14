import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import httpx
from fastapi.testclient import TestClient
from modelfoundry import sample_dataset, train_models
from modelops.api import create_app
from modelops.registry import Registry


class ServingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        data = sample_dataset()
        _, result = train_models(data,'churned','classification',['Logistic Regression'],.25)
        self.result = result
        self.registry = Registry(self.temp.name)
        self.version = self.registry.register(result['models']['Logistic Regression'], result['X_train'], {'task':'classification'})['version']
        self.rows = result['X_test'].head(4).to_dict(orient='records')

    def test_real_model_predictions_schema_drift_and_metrics_survive_restart(self):
        with TestClient(create_app(self.temp.name)) as client:
            response = client.post(f'/predict/{self.version}',json={'rows':self.rows})
            self.assertEqual(response.status_code,200,response.text)
            expected = self.result['models']['Logistic Regression'].predict(self.result['X_test'].head(4)).tolist()
            self.assertEqual(response.json()['predictions'],expected)
            shifted = [dict(row, monthly_charge=10000, plan='UNSEEN') for row in self.rows]
            self.assertEqual(client.post(f'/predict/{self.version}',json={'rows':shifted}).status_code,200)
            self.assertEqual(client.post(f'/predict/{self.version}',json={'rows':[{'wrong':1}]}).status_code,422)
            self.assertEqual(client.get('/monitor/not-a-version').status_code,404)
        with TestClient(create_app(self.temp.name)) as restarted:
            monitor = restarted.get(f'/monitor/{self.version}').json()
            self.assertEqual(monitor['requests'],2)
            self.assertEqual(monitor['rows'],8)
            self.assertGreater(monitor['latency_ms']['p95'],0)
            self.assertTrue(monitor['features']['monthly_charge']['drifted'])
            self.assertTrue(monitor['features']['plan']['drifted'])

    def test_auth_and_corrupt_artifact_fail_without_recording_success(self):
        with TestClient(create_app(self.temp.name,api_key='test-key')) as client:
            self.assertEqual(client.get('/health').status_code,200)
            self.assertEqual(client.get('/models').status_code,401)
            auth = {'Authorization':'Bearer test-key'}
            self.assertEqual(client.get('/models',headers=auth).status_code,200)
            artifact = Path(self.temp.name)/'artifacts'/f'{self.version}.joblib'
            artifact.write_bytes(b'corrupt')
            response = client.post(f'/predict/{self.version}',json={'rows':self.rows},headers=auth)
            self.assertEqual(response.status_code,503)
            self.assertEqual(self.registry.monitor(self.version)['requests'],0)

    def test_live_uvicorn_process_serves_predictions_over_http(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port = sock.getsockname()[1]
        env = dict(os.environ, MODELFOUNDRY_REGISTRY=self.temp.name)
        env.pop('MODELFOUNDRY_API_KEY',None)
        process = subprocess.Popen([sys.executable,'-m','uvicorn','modelops.api:app','--host','127.0.0.1','--port',str(port)],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic()+20
            while time.monotonic()<deadline:
                try:
                    response = httpx.get(base+'/health',timeout=1)
                    if response.status_code==200:
                        break
                except httpx.HTTPError:
                    pass
                if process.poll() is not None:
                    self.fail('Uvicorn exited before becoming ready')
                time.sleep(.1)
            else:
                self.fail('Uvicorn readiness timeout')
            response = httpx.post(base+f'/predict/{self.version}',json={'rows':self.rows},timeout=10)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(len(response.json()['predictions']),4)
            self.assertEqual(httpx.get(base+f'/monitor/{self.version}').json()['requests'],1)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


class RegressionPredictionIntegrationTests(unittest.TestCase):
    def test_regression_model_returns_numeric_predictions(self):
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        reference = pd.DataFrame({'x':[0.,1.,2.]})
        model = LinearRegression().fit(reference,[0.,2.,4.])
        with tempfile.TemporaryDirectory() as temp:
            registry = Registry(temp)
            version = registry.register(model,reference,{'task':'regression'})['version']
            with TestClient(create_app(temp)) as client:
                response = client.post(f'/predict/{version}',json={'rows':[{'x':1.5}]})
                self.assertEqual(response.status_code,200,response.text)
                self.assertAlmostEqual(response.json()['predictions'][0],3.)
            self.assertEqual(registry.monitor(version)['requests'],1)

    def test_overflow_prediction_is_rejected_before_recording_success(self):
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        reference = pd.DataFrame({'x':[0.,1.,2.]})
        model = LinearRegression().fit(reference,[0.,2.,4.])
        with tempfile.TemporaryDirectory() as temp:
            registry = Registry(temp)
            version = registry.register(model,reference,{'task':'regression'})['version']
            with TestClient(create_app(temp),raise_server_exceptions=False) as client:
                response = client.post(f'/predict/{version}',json={'rows':[{'x':1e308}]})
                self.assertEqual(response.status_code,422,response.text)
            self.assertEqual(registry.monitor(version)['requests'],0)
