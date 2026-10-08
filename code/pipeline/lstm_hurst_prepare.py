"""
lstm_hurst_prepare.py -- RETIRED.

This script belonged to the old per-company LSTM trainer, which is replaced by lstm_hurst_train.py: one script, scale-free
features, trading-day windows, a chronological train / validation / test split with an embargo, early stopping, calibration,
and a head-to-head against a logistic baseline (see the docstring of lstm_hurst_train.py for the evidence behind each change).

Models written by the old scripts are recognised by lstm_hurst_predictor.py and skipped (they must be retrained). Run instead:

    python lstm_hurst_train.py
"""
raise SystemExit(__doc__)
