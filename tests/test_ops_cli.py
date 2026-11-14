import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DeploymentCLIIntegrationTests(unittest.TestCase):
    def test_demo_trains_registers_serves_and_exports_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'demo'
            result = subprocess.run([sys.executable,'-m','modelops.demo','--out',str(out)],capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            evidence = json.loads((out/'evidence.json').read_text())
            self.assertEqual(evidence['versions_registered'],2)
            self.assertTrue(evidence['predictions_match'])
            self.assertTrue(evidence['drift_detected'])
            self.assertTrue(evidence['metrics_survive_restart'])
            self.assertTrue((out/'report.html').is_file())
            self.assertEqual(evidence['model_calls'],0)
            repeated = subprocess.run([sys.executable,'-m','modelops.demo','--out',str(out)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(repeated.returncode,0)

    def test_training_cli_registers_a_real_model_from_csv(self):
        from modelfoundry import sample_dataset
        from modelops.registry import Registry
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'sample.csv'
            sample_dataset().to_csv(source,index=False)
            registry = root/'registry'
            result = subprocess.run([sys.executable,'-m','modelops.cli','--csv',str(source),
                '--target','churned','--task','classification','--algorithm','Logistic Regression',
                '--registry',str(registry)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            manifest = json.loads(result.stdout)
            self.assertEqual(manifest['metadata']['algorithm'],'Logistic Regression')
            self.assertEqual(manifest['reference_rows'],315)
            model,_,_ = Registry(registry).load(manifest['version'])
            self.assertEqual(len(model.predict(sample_dataset().drop(columns=['churned']).head(3))),3)
