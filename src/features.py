"""
src/features.py

Builds the customer-level feature table used for churn modeling:
- Core RFM features (recency, frequency, monetary)
- Trend/momentum features (recent vs. prior activity)
- Behavioral features (category breadth, cancellation/return behavior)
- Churn label (based on CHURN_WINDOW_DAYS from config)

All features are computed strictly from data at or before the reference date,
to avoid target leakage.
"""

import numpy as np
import pandas as pd

from src.config import CHURN_WINDOW_DAYS


def compute_rfm_features(purchases: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Core RFM features per customer, computed from purchase (non-cancellation) rows."""
    df = purchases.copy()
    df["line_total"] = df["Quantity"] * df["Price"]

    rfm = (
        df.groupby("Customer ID")
        .agg(
            last_purchase_date=("InvoiceDate", "max"),
            first_purchase_date=("InvoiceDate", "min"),
            frequency=("Invoice", "nunique"),
            monetary_total=("line_total", "sum"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (reference_date - rfm["last_purchase_date"]).dt.days
    rfm["tenure_days"] = (reference_date - rfm["first_purchase_date"]).dt.days
    rfm["monetary_avg"] = rfm["monetary_total"] / rfm["frequency"]

    # spend variability per order (std of order-level totals)
    order_totals = df.groupby(["Customer ID", "Invoice"])["line_total"].sum().reset_index()
    monetary_std = order_totals.groupby("Customer ID")["line_total"].std().reset_index(
        name="monetary_std"
    )
    rfm = rfm.merge(monetary_std, on="Customer ID", how="left")
    rfm["monetary_std"] = rfm["monetary_std"].fillna(0)

    return rfm


def compute_trend_features(
    purchases: pd.DataFrame, reference_date: pd.Timestamp, window_days: int = 90
) -> pd.DataFrame:
    """
    Compares recent activity (last `window_days`) to the prior period of the
    same length, to capture momentum rather than just static RFM snapshots.
    """
    df = purchases.copy()
    df["line_total"] = df["Quantity"] * df["Price"]

    recent_start = reference_date - pd.Timedelta(days=window_days)
    prior_start = reference_date - pd.Timedelta(days=2 * window_days)

    recent = df[df["InvoiceDate"] > recent_start]
    prior = df[(df["InvoiceDate"] > prior_start) & (df["InvoiceDate"] <= recent_start)]

    def agg_period(data: pd.DataFrame, suffix: str) -> pd.DataFrame:
        out = (
            data.groupby("Customer ID")
            .agg(**{
                f"orders_{suffix}": ("Invoice", "nunique"),
                f"spend_{suffix}": ("line_total", "sum"),
            })
            .reset_index()
        )
        return out

    recent_agg = agg_period(recent, "recent")
    prior_agg = agg_period(prior, "prior")

    trend = recent_agg.merge(prior_agg, on="Customer ID", how="outer").fillna(0)

    # ratio-based trend (add 1 to denominator to avoid divide-by-zero)
    trend["order_frequency_trend"] = trend["orders_recent"] / (trend["orders_prior"] + 1)
    trend["spend_trend"] = trend["spend_recent"] / (trend["spend_prior"] + 1)

    return trend[["Customer ID", "orders_recent", "orders_prior",
                   "spend_recent", "spend_prior",
                   "order_frequency_trend", "spend_trend"]]


def compute_behavioral_features(purchases: pd.DataFrame, cancellations: pd.DataFrame) -> pd.DataFrame:
    """Category breadth and return/cancellation behavior per customer."""
    breadth = (
        purchases.groupby("Customer ID")["StockCode"]
        .nunique()
        .reset_index(name="n_distinct_products")
    )

    country = (
        purchases.groupby("Customer ID")["Country"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index(name="country")
    )

    total_orders = purchases.groupby("Customer ID")["Invoice"].nunique().reset_index(
        name="_total_orders"
    )
    cancel_orders = (
        cancellations.groupby("Customer ID")["Invoice"]
        .nunique()
        .reset_index(name="_cancel_orders")
    )

    returns = total_orders.merge(cancel_orders, on="Customer ID", how="left")
    returns["_cancel_orders"] = returns["_cancel_orders"].fillna(0)
    returns["has_returned"] = returns["_cancel_orders"] > 0
    returns["return_rate"] = returns["_cancel_orders"] / (
        returns["_total_orders"] + returns["_cancel_orders"]
    )

    behavioral = breadth.merge(country, on="Customer ID", how="left").merge(
        returns[["Customer ID", "has_returned", "return_rate"]], on="Customer ID", how="left"
    )
    behavioral["has_returned"] = behavioral["has_returned"].fillna(False)
    behavioral["return_rate"] = behavioral["return_rate"].fillna(0)

    return behavioral


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform heavily right-skewed features (frequency, monetary) for modeling."""
    df = df.copy()
    df["log_frequency"] = np.log1p(df["frequency"])
    df["log_monetary_total"] = np.log1p(df["monetary_total"])
    return df


def build_feature_table(
    purchases: pd.DataFrame,
    cancellations: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Full pipeline: builds the customer-level feature table with RFM, trend,
    behavioral features, log transforms, and the churn label.
    """
    rfm = compute_rfm_features(purchases, reference_date)
    trend = compute_trend_features(purchases, reference_date)
    behavioral = compute_behavioral_features(purchases, cancellations)

    features = (
        rfm.merge(trend, on="Customer ID", how="left")
        .merge(behavioral, on="Customer ID", how="left")
    )

    trend_cols = ["orders_recent", "orders_prior", "spend_recent", "spend_prior",
                  "order_frequency_trend", "spend_trend"]
    features[trend_cols] = features[trend_cols].fillna(0)

    features = add_log_transforms(features)

    # target: churn label based on recency vs. the empirically-derived window
    features["is_churned"] = (features["recency_days"] > CHURN_WINDOW_DAYS).astype(int)

    return features