import unittest
from modelfoundry import sample_dataset, train_models


class TrainingReferenceTests(unittest.TestCase):
    def test_reference_is_exactly_training_split_and_excludes_holdout(self):
        data = sample_dataset()
        _, result = train_models(data, 'churned', 'classification', ['Logistic Regression'], .25)
        reference = result['X_train']
        self.assertEqual(len(reference), result['train_rows'])
        self.assertTrue(set(reference.index).isdisjoint(result['X_test'].index))
        self.assertEqual(set(reference.index) | set(result['X_test'].index), set(data.index))
        self.assertNotIn('churned', reference.columns)
