"""
mcs_pipeline
============
Modular pipeline that turns raw 5-year stock history files into four
progressively richer datasets:

    1. MCS_raw.csv     - calendar-complete raw spine (price/volume + calendar flags)
    2. MCS_report.csv  - MCS_raw + sentiment/macro/fundamental features + REAL health score
    3. MCS_predict.csv - MCS_raw + enrichment features + XGBoost volume prediction
    4. MCS_health.csv  - MCS_predict + PREDICTED health score (uses XGBoost output)

Run `python -m mcs_pipeline.pipeline` (or `python pipeline.py` from inside the
package folder) to execute the full pipeline end to end.
"""
