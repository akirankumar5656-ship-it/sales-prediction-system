import math
import os
from pathlib import Path
import tempfile
import unittest

import config


class Workflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.original_db = config.DATABASE_PATH
        config.DATABASE_PATH = str(Path(cls.tmp.name) / 'test.db')
        import app
        cls.module = app
        app.app.config.update(TESTING=True)

    @classmethod
    def tearDownClass(cls):
        config.DATABASE_PATH = cls.original_db
        cls.tmp.cleanup()

    def setUp(self):
        self.client = self.module.app.test_client()

    def test_pages_and_figures(self):
        for path in ['/', '/predict', '/analytics', '/dashboard', '/performance', '/history', '/login', '/register']:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)
        for figure in Path(config.REPORTS_EVAL_FIGURES_DIR).glob('*.png'):
            with self.client.get('/figures/evaluation/' + figure.name) as response:
                self.assertEqual(response.status_code, 200)

    def test_real_prediction_and_invalid_inputs(self):
        self.assertIsNotNone(self.module.prediction_service, self.module.MODEL_LOAD_ERROR)
        response = self.client.post('/predict', data={'TV': '100', 'Radio': '20', 'Newspaper': '10'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction Result', response.data)
        from database import db_operations as db
        before = db.get_dashboard_stats()['total_predictions']
        for value in ['', 'abc', '-1', 'nan', 'inf']:
            with self.subTest(value=value):
                response = self.client.post('/predict', data={'TV': value, 'Radio': '20', 'Newspaper': '10'}, follow_redirects=True)
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Invalid prediction input', response.data)
        self.assertEqual(db.get_dashboard_stats()['total_predictions'], before)

    def test_accounts_and_history_isolation(self):
        from database import db_operations as db
        response = self.client.post('/register', data={'username': 'alice', 'password': 'example-password', 'confirm_password': 'example-password'}, follow_redirects=True)
        self.assertIn(b'Account created', response.data)
        self.client.post('/predict', data={'TV': '123.456', 'Radio': '20', 'Newspaper': '10'})
        self.assertIn(b'123.456', self.client.get('/history').data)
        self.assertIn(b'Advertising', self.client.get('/dashboard').data)
        guest = self.module.app.test_client()
        self.assertNotIn(b'123.456', guest.get('/history').data)
        alice = db.verify_user('alice', 'example-password')
        self.assertIsNotNone(alice)
        self.assertIsNone(db.verify_user('alice', 'wrong'))
        self.assertTrue(all(r['user_id'] is None for r in db.get_prediction_history()))
        self.client.get('/logout')
        self.client.post('/login', data={'username': 'alice', 'password': 'example-password'})
        self.assertIn(b'123.456', self.client.get('/history').data)

    def test_model_independent_of_working_directory(self):
        from src.predict import PredictionService
        previous = os.getcwd()
        try:
            os.chdir(self.tmp.name)
            result = PredictionService().predict({'TV': 100, 'Radio': 20, 'Newspaper': 10})
            self.assertTrue(math.isfinite(result))
            self.assertGreater(result, 0)
        finally:
            os.chdir(previous)

    def test_saved_metrics_match_evaluation(self):
        import pandas as pd
        comparison = pd.read_csv(config.MODEL_COMPARISON_PATH).sort_values('Model').reset_index(drop=True)
        evaluation = pd.read_csv(config.EVALUATION_METRICS_PATH).sort_values('Model').reset_index(drop=True)
        pd.testing.assert_frame_equal(comparison, evaluation)

    def test_history_export_keeps_account_records_private(self):
        import csv
        import io
        member = self.module.app.test_client()
        member.post('/register', data={'username': 'export-user', 'password': 'test-password',
                                      'confirm_password': 'test-password'})
        member.post('/predict', data={'TV': '141.234', 'Radio': '12', 'Newspaper': '8'})
        response = member.get('/history/export')
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment;', response.headers['Content-Disposition'])
        rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
        self.assertEqual(len(rows), 2)
        self.assertIn('141.234', rows[1][1])
        self.assertNotIn(b'141.234', self.client.get('/history/export').data)

    def test_advertising_insights_and_preserved_inputs(self):
        from src import analytics
        stats = analytics.compute_advertising_stats(self.module.prediction_service.history)
        self.assertEqual(stats['rows'], 200)
        self.assertAlmostEqual(sum(channel['share'] for channel in stats['channels']), 100)
        response = self.client.post('/predict', data={'TV': '9999', 'Radio': '20', 'Newspaper': '10'})
        self.assertIn(b'outside the observed range', response.data)
        self.assertIn(b'Total advertising spend', response.data)
        self.assertNotIn(b'<title>Prediction Result<a', response.data)
        response = self.client.post('/predict', data={'TV': 'abc', 'Radio': '22.5', 'Newspaper': '10'}, follow_redirects=True)
        self.assertIn(b'value="22.5"', response.data)
        page = self.client.get('/analytics').data
        self.assertIn(b'Where campaigns spend', page)
        self.assertNotIn(b'No Category column', page)
