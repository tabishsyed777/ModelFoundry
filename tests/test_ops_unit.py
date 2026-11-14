import copy
import math
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from modelfoundry import build_pipeline
from modelops.profile import build_profile, compare_profile, validate_rows
from modelops.registry import Registry


def fitted():
    frame = pd.DataFrame({'age':[20,25,30,35,40,45,50,55], 'plan':['A','B']*4})
    model = build_pipeline(frame, LogisticRegression()).fit(frame, [0,0,0,1,0,1,1,1])
    return frame, model


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.frame, _ = fitted()
        self.profile = build_profile(self.frame)

    def test_same_distribution_has_zero_drift_and_shift_is_flagged(self):
        baseline = compare_profile(self.profile, self.frame)
        self.assertTrue(all(item['distance'] == 0 for item in baseline.values()))
        shifted = self.frame.assign(age=999, plan='unseen')
        self.assertTrue(all(item['drifted'] for item in compare_profile(self.profile, shifted).values()))

    def test_strict_schema_rejects_missing_extra_wrong_type_and_nonfinite(self):
        valid = [{'age':30, 'plan':'A'}, {'age':None,'plan':None}]
        self.assertEqual(len(validate_rows(self.profile, valid)), 2)
        for rows in [[], [{'age':3}], [{'age':3,'plan':'A','extra':4}],
                     [{'age':'3','plan':'A'}], [{'age':True,'plan':'A'}],
                     [{'age':math.inf,'plan':'A'}], [{'age':10**400,'plan':'A'}], [{'age':3,'plan':{}}]]:
            with self.assertRaises(ValueError):
                validate_rows(self.profile, rows)

    def test_unknown_and_missing_categories_use_distinct_buckets(self):
        result = compare_profile(self.profile, pd.DataFrame({'age':[None,None], 'plan':[None,'unseen']}))
        self.assertEqual(result['plan']['counts'][-2:], [1,1])
        self.assertEqual(sum(result['age']['counts']),2)

    def test_boolean_schema_is_preserved(self):
        profile = build_profile(pd.DataFrame({'active':[True,False,True]}))
        self.assertEqual(len(validate_rows(profile,[{'active':False}])),1)
        with self.assertRaises(ValueError):
            validate_rows(profile,[{'active':1}])


class RegistryTests(unittest.TestCase):
    def test_immutable_versions_artifact_integrity_and_restart(self):
        frame, model = fitted()
        with tempfile.TemporaryDirectory() as temp:
            registry = Registry(temp)
            first = registry.register(model, frame, {'algorithm':'Logistic Regression'})
            second = registry.register(model, frame, {'algorithm':'Logistic Regression'})
            self.assertNotEqual(first['version'],second['version'])
            restarted = Registry(temp)
            self.assertEqual(len(restarted.list_versions()),2)
            loaded, metadata, _ = restarted.load(first['version'])
            self.assertEqual(loaded.predict(frame).tolist(), model.predict(frame).tolist())
            self.assertEqual(metadata['metadata']['algorithm'],'Logistic Regression')
            artifact = Path(temp)/'artifacts'/f"{first['version']}.joblib"
            artifact.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                restarted.load(first['version'])

    def test_metrics_are_persistent_and_version_scoped(self):
        frame, model = fitted()
        with tempfile.TemporaryDirectory() as temp:
            registry = Registry(temp)
            first = registry.register(model,frame,{})
            second = registry.register(model,frame,{})
            profile = build_profile(frame)
            counts = {name:item['counts'] for name,item in compare_profile(profile,frame).items()}
            registry.record(first['version'],counts,12.5,len(frame))
            registry.record(first['version'],counts,22.5,len(frame))
            monitor = Registry(temp).monitor(first['version'])
            self.assertEqual(monitor['requests'],2)
            self.assertEqual(monitor['rows'],16)
            self.assertEqual(monitor['latency_ms']['mean'],17.5)
            self.assertTrue(all(f['distance']==0 for f in monitor['features'].values()))
            self.assertEqual(registry.monitor(second['version'])['requests'],0)
