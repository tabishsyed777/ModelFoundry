"""Opt-in real container integration test. CI builds the image before running this."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import httpx
from modelfoundry import sample_dataset

IMAGE = os.getenv('MODELFOUNDRY_TEST_IMAGE')


@unittest.skipUnless(IMAGE,'Set MODELFOUNDRY_TEST_IMAGE after building the Docker image')
class ContainerIntegrationTests(unittest.TestCase):
    def test_train_register_and_serve_in_the_same_image(self):
        inspected = subprocess.run(['docker','image','inspect',IMAGE],capture_output=True,text=True,timeout=10)
        self.assertEqual(inspected.returncode,0,inspected.stderr)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample_dataset().to_csv(root/'sample.csv',index=False)
            common = ['--user',f'{os.getuid()}:{os.getgid()}','-v',f'{root}:/data']
            trained = subprocess.run(['docker','run','--rm',*common,IMAGE,'python','-m','modelops.cli',
                '--csv','/data/sample.csv','--target','churned','--task','classification',
                '--algorithm','Logistic Regression','--registry','/data/registry'],capture_output=True,text=True,timeout=60)
            self.assertEqual(trained.returncode,0,trained.stdout+trained.stderr)
            version = json.loads(trained.stdout)['version']
            started = subprocess.run(['docker','run','--rm','-d',*common,'-p','127.0.0.1::8000',
                '-e','MODELFOUNDRY_REGISTRY=/data/registry','-e','MODELFOUNDRY_API_KEY=container-test-key',IMAGE],capture_output=True,text=True,timeout=15)
            self.assertEqual(started.returncode,0,started.stderr)
            container = started.stdout.strip()
            try:
                port = subprocess.check_output(['docker','port',container,'8000/tcp'],text=True).strip()
                base = 'http://'+port
                deadline = time.monotonic()+30
                while time.monotonic()<deadline:
                    try:
                        if httpx.get(base+'/health',timeout=1).status_code==200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.2)
                else:
                    self.fail('Container health timeout')
                self.assertEqual(httpx.get(base+'/models').status_code,401)
                auth = {'Authorization':'Bearer container-test-key'}
                rows = sample_dataset().drop(columns=['churned']).head(3).to_dict(orient='records')
                response = httpx.post(base+f'/predict/{version}',json={'rows':rows},headers=auth,timeout=10)
                self.assertEqual(response.status_code,200,response.text)
                self.assertEqual(len(response.json()['predictions']),3)
                monitor = httpx.get(base+f'/monitor/{version}',headers=auth).json()
                self.assertEqual(monitor['requests'],1)
                self.assertEqual(monitor['rows'],3)
            finally:
                subprocess.run(['docker','rm','-f',container],capture_output=True,timeout=15)
