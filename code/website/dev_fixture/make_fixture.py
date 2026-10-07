"""SYNTHETIC test data for the UI only. NOT real backend output.
Takes testing/sample_data/mock_test_real_data.csv and adds the columns stage 6 would add, using the
pipeline's own scoring code. Predictions/breakout are random (no trained models here)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from pipeline import config
from pipeline.fuzzy_system import BusinessHealthFuzzySystem
from pipeline.support_resistance import add_support_resistance

rng = np.random.default_rng(42)
df = pd.read_csv(ROOT / "testing/sample_data/mock_test_real_data.csv", parse_dates=["Date"]).drop(columns=["market_cap"])
df[config.PREDICTION_COLUMN] = (df["target_volume_T_plus_1"] * rng.lognormal(0, .25, len(df))).round()
df["hurst_exponent"] = rng.uniform(.4, .75, len(df))
df["breakout_probability"] = np.where(df.symbol.isin(["GIAA.JK", "HATM.JK", "WBSA.JK"]), np.nan, rng.uniform(.05, .8, len(df)))
try: df = add_support_resistance(df)
except Exception as e: print("S/R skipped:", e)
df["sentiment_is_mock"] = True          # the sentiment in the sample file is generated, not BERT output
# score_frame = the same single call pipeline.py (stage 6) and website/server.py use
df = BusinessHealthFuzzySystem().score_frame(df)
out = Path(__file__).parent / "MCS_health.csv"; df.to_csv(out, index=False); print("wrote", out, df.shape)
