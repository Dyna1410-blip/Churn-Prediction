"""
src/config.py

Central place for constants and assumptions used across the project.
Documenting these here (rather than scattering magic numbers through
notebooks/scripts) makes assumptions explicit and easy to revisit.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --- Churn definition ---
# 75 days sits between the 75th (61 days) and 80th (76 days) percentile of
# natural reorder gaps in this customer base — chosen to flag customers who
# have moved meaningfully beyond typical reorder behavior, while still
# leaving enough lead time for a retention intervention to be actionable.
CHURN_WINDOW_DAYS = 75

# --- Business assumptions (for ROI/impact calculations later) ---
RETENTION_CAMPAIGN_COST_PER_CUSTOMER = 3.0   # e.g., discount/email cost, in GBP
RETENTION_CAMPAIGN_SUCCESS_RATE = 0.15       # assumed % of targeted at-risk customers who convert