"""
app/streamlit_app.py

Interactive dashboard for the e-commerce churn prediction project.

Two views:
1. Customer Risk Explorer — browse/filter customers by predicted churn risk
2. ROI Simulator — interactively explore when the model-targeted campaign
   beats a simple blanket campaign, as campaign cost varies

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROCESSED_DATA_DIR,
    RETENTION_CAMPAIGN_COST_PER_CUSTOMER,
    RETENTION_CAMPAIGN_SUCCESS_RATE,
    CHURN_WINDOW_DAYS,
)

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "customer_predictions.parquet"
    return pd.read_parquet(path)


def main():
    st.title("E-Commerce Customer Churn Dashboard")
    st.caption(
        f"Churn defined as no purchase within {CHURN_WINDOW_DAYS} days of a customer's "
        "last order — a threshold derived from this customer base's typical reorder gaps."
    )

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(
            "No predictions file found. Run 04_modeling.ipynb through the "
            "prediction-saving cell first to generate data/processed/customer_predictions.parquet."
        )
        return

    tab1, tab2 = st.tabs(["Customer Risk Explorer", "ROI Simulator"])

    # --- Tab 1: Customer Risk Explorer ---
    with tab1:
        st.subheader("Customer Risk Explorer")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", f"{len(df):,}")
        col1.metric("Predicted At-Risk", f"{df['predicted_churn'].sum():,}")
        col2.metric("Avg Churn Probability", f"{df['churn_probability'].mean():.1%}")
        col3.metric("Avg Customer Value", f"£{df['monetary_total'].mean():,.0f}")

        st.divider()

        min_prob = st.slider("Minimum churn probability", 0.0, 1.0, 0.5, 0.05)

        filtered = df[df["churn_probability"] >= min_prob].sort_values(
            "churn_probability", ascending=False
        )

        st.write(f"Showing {len(filtered):,} customers at or above {min_prob:.0%} churn probability")

        display_cols = [
            "Customer ID", "churn_probability", "frequency", "monetary_total",
            "tenure_days", "recency_days", "return_rate", "country",
        ]
        st.dataframe(
            filtered[display_cols].style.format({
                "churn_probability": "{:.1%}",
                "monetary_total": "£{:,.0f}",
                "return_rate": "{:.1%}",
            }),
            use_container_width=True,
            height=450,
        )

    # --- Tab 2: ROI Simulator ---
    with tab2:
        st.subheader("Is using the model really beneficial ?")
        st.markdown(
            """
            This tool compares two strategies for a customer retention campaign
            (e.g. a "we miss you" email or discount offer):

            - **🎯 Model-Targeted** — only contact the customers our model predicts are at risk of churning
            - **📢 Blind Campaign** — contact *every* customer, no filtering

            Adjust the sliders below to see which strategy makes more sense under
            different assumptions.
            """
        )

        col_a, col_b = st.columns(2)
        with col_a:
            campaign_cost = st.slider(
                "💷 Cost to contact one customer (£)",
                1.0, 60.0, float(RETENTION_CAMPAIGN_COST_PER_CUSTOMER), 0.5,
                help="E.g. £3 for a simple email, £20-50 for a phone call or a bigger discount offer.",
            )
        with col_b:
            success_rate = st.slider(
                "✅ % of contacted at-risk customers who actually come back",
                1, 50, int(RETENTION_CAMPAIGN_SUCCESS_RATE * 100), 1,
                help="Not everyone who gets an email returns — this is the assumed conversion rate.",
            ) / 100

        avg_order_value = df["monetary_avg"].mean()

        targeted = df[df["predicted_churn"] == 1]
        actual_churners_targeted = targeted[targeted["is_churned"] == 1]

        n_targeted = len(targeted)
        n_saved_targeted = len(actual_churners_targeted) * success_rate
        revenue_targeted = n_saved_targeted * avg_order_value
        cost_targeted = n_targeted * campaign_cost
        net_targeted = revenue_targeted - cost_targeted

        n_total = len(df)
        n_actual_churners = df["is_churned"].sum()
        n_saved_blanket = n_actual_churners * success_rate
        revenue_blanket = n_saved_blanket * avg_order_value
        cost_blanket = n_total * campaign_cost
        net_blanket = revenue_blanket - cost_blanket

        model_wins = net_targeted > net_blanket
        advantage = abs(net_targeted - net_blanket)

        st.divider()

        # --- Plain-language verdict, front and center ---
        if model_wins:
            st.success(
                f"### 🎯 At £{campaign_cost:.0f} per customer, **using the model saves £{advantage:,.0f}** "
                f"compared to emailing everyone.\n\n"
                f"At this cost, wasting money contacting customers who were never going to leave adds up — "
                f"so it pays to be selective and only contact the **{n_targeted:,} customers** "
                f"the model flags as at-risk."
            )
        else:
            st.info(
                f"### 📢 At £{campaign_cost:.0f} per customer, **emailing everyone saves £{advantage:,.0f}** "
                f"compared to using the model.\n\n"
                f"At this low a cost, contacting people unnecessarily barely matters — "
                f"it's cheaper to just reach out to all **{n_total:,} customers** than to build "
                f"and maintain a targeting model."
            )

        st.divider()

        # --- Side-by-side comparison, in plain terms ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Model-Targeted")
            st.metric("Customers contacted", f"{n_targeted:,}")
            st.metric("Total campaign cost", f"£{cost_targeted:,.0f}")
            st.metric("Revenue recovered", f"£{revenue_targeted:,.0f}")
            st.metric("Net result", f"£{net_targeted:,.0f}")

        with col2:
            st.markdown("#### 📢 Blanket Campaign")
            st.metric("Customers contacted", f"{n_total:,}")
            st.metric("Total campaign cost", f"£{cost_blanket:,.0f}")
            st.metric("Revenue recovered", f"£{revenue_blanket:,.0f}")
            st.metric("Net result", f"£{net_blanket:,.0f}")

        st.divider()

        # --- Break-even chart, simplified explanation ---
        st.markdown("#### 📈 When does each strategy win?")
        st.caption(
            "This chart shows the same comparison across a full range of campaign costs. "
            "Where the two lines cross is the 'break-even point' — the cost at which it "
            "starts being worth it to use the model instead of contacting everyone."
        )

        costs = list(range(1, 61))
        model_values, blanket_values = [], []
        for c in costs:
            mv = (len(actual_churners_targeted) * success_rate * avg_order_value) - (n_targeted * c)
            bv = (n_actual_churners * success_rate * avg_order_value) - (n_total * c)
            model_values.append(mv)
            blanket_values.append(bv)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(costs, model_values, label="🎯 Model-Targeted", linewidth=2.5, color="#2563eb")
        ax.plot(costs, blanket_values, label="📢 Blanket Campaign", linewidth=2.5, color="#f97316")
        ax.axvline(campaign_cost, color="gray", linestyle="--", alpha=0.7,
                   label=f"Current setting (£{campaign_cost:.0f})")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Cost to contact one customer (£)")
        ax.set_ylabel("Net value to the business (£)")
        ax.set_title("Which strategy makes more money, at each campaign cost?")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.caption(
            "💡 **Takeaway:** cheap campaigns (like a generic email) don't need a targeting model — "
            "just contact everyone. Expensive campaigns (like a phone call or a big discount) "
            "benefit from being selective, since wasting money on customers who weren't leaving "
            "anyway starts to really add up."
        )


if __name__ == "__main__":
    main()