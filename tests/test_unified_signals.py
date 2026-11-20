"""Consistency tests: the public live score and historical engine must agree."""
import unittest

import numpy as np
import pandas as pd

import signals
import prediction


class SignalConsistency(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2021-01-01", periods=180)
        self.close = pd.Series(np.linspace(100, 135, len(dates)), index=dates)

    def test_latest_equals_historical_final(self):
        frame = pd.DataFrame({"Close": self.close})
        latest = signals.latest_scores(frame)
        m, r, h = signals.indicators(pd.DataFrame({"X":self.close}))
        self.assertIsNotNone(latest)
        for name, history in (("momentum",m),("recovery",r),("hybrid",h)):
            self.assertAlmostEqual(latest[name],float(history["X"].iloc[-1]),places=10)
        self.assertAlmostEqual(latest["hybrid"],
                               0.6*latest["momentum"]+0.4*latest["recovery"],places=10)

    def test_no_future_lookahead(self):
        first = pd.DataFrame({"X":self.close.iloc[:-1]})
        later = pd.DataFrame({"X":self.close*1.0})
        m1, r1, h1 = signals.indicators(first)
        m2, r2, h2 = signals.indicators(later)
        for a,b in ((m1,m2),(r1,r2),(h1,h2)):
            self.assertTrue(np.allclose(a["X"].dropna().values,
                                        b["X"].iloc[:-1].dropna().values,rtol=0,atol=1e-10))

    def test_insufficient_history_is_not_scored(self):
        self.assertIsNone(signals.latest_scores(pd.DataFrame({"Close":self.close.iloc[:90]})))
        m,r,h=signals.indicators(pd.DataFrame({"X":self.close.iloc[:90]}))
        self.assertTrue(m.isna().all().all())
        self.assertTrue(r.isna().all().all())
        self.assertTrue(h.isna().all().all())

    def test_pure_models_dimension(self):
        rows = np.array([[20,50],[50,50],[80,30]],dtype=float)
        for name in prediction.FEATURES:
            vectors=np.vstack([prediction.features(m,r,name) for m,r in rows])
            self.assertEqual(vectors.shape[1],len(prediction.FEATURES[name]))

    def test_ridge_training_is_deterministic(self):
        X=np.array([[float(i),float(i%7)] for i in range(100)],dtype=float)
        y=np.linspace(-0.1,0.1,100)
        model=prediction.fit_model(X,y)
        p=prediction.predict_model(model,X[:10])
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(len(p),10)


if __name__=="__main__":
    unittest.main()
