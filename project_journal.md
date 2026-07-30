# Project Journal — E-Commerce Churn Prediction

A detailed technical log of every step, the exact functions used, and the reasoning behind each decision. Written for my own future reference (interview prep, resume writing, reuse in future projects).

---

## Stage 1: Data Ingestion

**Goal:** Download the dataset programmatically and load it into pandas.

| Function | Library | What it did |
|---|---|---|
| `kagglehub.dataset_download(slug)` | `kagglehub` | Downloads a Kaggle dataset via API, caches it locally, returns the cache path |
| `load_dotenv()` | `python-dotenv` | Loads `KAGGLE_USERNAME`/`KAGGLE_KEY` from a `.env` file into environment variables |
| `Path.glob("*.csv")` | `pathlib` | Finds all CSV files in a directory matching a pattern |
| `shutil.copy2(src, dst)` | `shutil` | Copies a file, preserving metadata, from the kagglehub cache into `data/raw/` |
| `pd.read_csv(path, encoding="ISO-8859-1")` | `pandas` | Loads the CSV; the non-default encoding was needed because the dataset contained special characters (e.g. £) that broke UTF-8 decoding |

**Key decision:** used `ISO-8859-1` encoding instead of default UTF-8 — learned this from a `UnicodeDecodeError` when first loading the file.

---

## Stage 2: Data Cleaning

**Goal:** Remove unusable rows, handle known data quirks (missing IDs, cancellations, bad prices).

| Function | What it did |
|---|---|
| `df["Customer ID"].notna()` | Boolean mask to find rows with a real customer ID |
| `df[mask]` | Filters the DataFrame to keep only rows matching a boolean condition |
| `df["Invoice"].astype(str).str.startswith("C")` | Identifies cancellation invoices (Olist/Online Retail convention: invoice numbers starting with "C" are cancellations) |
| `df["Price"] > 0` | Filters out zero/negative price rows (data artifacts, not real transactions) |
| `pd.to_datetime(df["InvoiceDate"])` | Converts a string date column into a real `datetime64` type, enabling date arithmetic later |
| `df["Customer ID"].astype(int).astype(str)` | Fixed a dtype issue — `Customer ID` loaded as `float64` (e.g. `17850.0`) because of prior NaNs; converting to int then str avoided float-precision issues when grouping |

**Key decision:** kept cancellation rows (flagged via a new `is_cancellation` boolean column) instead of dropping them, so return behavior could later become a feature (`return_rate`, `has_returned`) rather than being thrown away.

---

## Stage 3: Exploratory Data Analysis

**Goal:** Understand repeat-purchase behavior, cohort retention, and RFM distribution shapes — before deciding on a churn definition.

| Function | What it did |
|---|---|
| `df.groupby("Customer ID")["Invoice"].nunique()` | Counted the number of distinct orders per customer (repeat-purchase check) |
| `df.set_index("InvoiceDate").resample("ME").size()` | Resampled transaction-level data into monthly counts — `"ME"` = month-end frequency. Used to check order volume trends over time. |
| `df["InvoiceDate"].dt.to_period("M")` | Converted a datetime into a year-month period, used for cohort grouping |
| `df.groupby("Customer ID")["order_month"].min()` | Found each customer's first purchase month (their acquisition cohort) |
| `df.pivot(index=..., columns=..., values=...)` | Reshaped long-format cohort counts into a wide cohort-by-month matrix |
| `pivot_table.divide(cohort_size, axis=0)` | Converted raw cohort counts into retention *percentages*, dividing each row by that cohort's starting size |
| `sns.histplot()` | Plotted distribution shapes (recency, frequency, monetary) |
| `sns.heatmap(annot=True, fmt=".0%")` | Visualized the cohort retention matrix as a color-coded heatmap with percentage labels |
| `groupby(...).diff()` | Computed the gap in days between a customer's consecutive orders, used to empirically derive the churn window (75 days) from percentile analysis of these gaps |
| `series.quantile(pct)` | Calculated specific percentiles (e.g., 75th, 90th) of the gap distribution to justify the churn threshold choice |

**Key decision:** derived the 75-day churn window empirically from the 75th-80th percentile of real reorder gaps, rather than guessing a round number like 30/60/90 days.

---

## Stage 4: Feature Engineering

**Goal:** Build a customer-level feature table: RFM, trend/momentum, and behavioral features.

| Function | What it did |
|---|---|
| `df.groupby("Customer ID").agg(name=(col, func), ...)` | Named aggregation — computed multiple summary stats per customer in one call (e.g., `frequency=("Invoice", "nunique")`) |
| `(reference_date - df["last_purchase_date"]).dt.days` | Computed recency in days by subtracting two datetime columns and extracting `.dt.days` |
| `groupby(...)["line_total"].std()` | Computed spend variability (`monetary_std`) per customer |
| `pd.Timedelta(days=n)` | Created a time offset used to define window boundaries (e.g., "90 days before the reference date") |
| `df[(df["InvoiceDate"] > start) & (df["InvoiceDate"] <= end)]` | Filtered transactions into a specific date window for trend feature calculation |
| `merge(..., how="outer").fillna(0)` | Combined "recent" and "prior" window aggregates; outer join + fillna(0) handled customers with zero activity in one or both windows |
| `np.log1p(x)` | Log-transformed heavily right-skewed features (`frequency`, `monetary_total`) — `log1p` (log(1+x)) avoids errors on zero values, unlike plain `log` |
| `pd.get_dummies(df, columns=[...], drop_first=True)` | One-hot encoded the `country_grouped` categorical column; `drop_first=True` avoided the "dummy variable trap" (perfect multicollinearity) by dropping one baseline category |
| `series.where(condition, other=...)` | Grouped rare countries into "Other" — kept the value where the condition was true, replaced it otherwise |
| `.astype(int)` on a boolean churn condition | Converted the True/False churn label into 1/0 for modeling |

**Key decision & mistake caught:** the first version of the trend features (`orders_recent` = activity in the last 90 days) directly leaked the churn label, since "recent activity" and "not churned" were almost the same thing by construction. Fixed by shifting both trend windows to start *after* the 75-day churn threshold (`leakage_gap_days` parameter), so trend features reflect genuine historical behavior with no overlap with the label's own definition window.

---

## Stage 5: Modeling

**Goal:** Train and compare models; tune the decision threshold for the business context.

| Function | Library | What it did |
|---|---|---|
| `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` | `sklearn.model_selection` | Split data into train/test sets; `stratify=y` preserved the same churn ratio in both sets |
| `StandardScaler().fit_transform(X_train)` | `sklearn.preprocessing` | Scaled features to mean 0, std 1 — necessary for Logistic Regression (distance/gradient-based), not needed for tree models |
| `.transform(X_test)` (not `.fit_transform`) | `sklearn.preprocessing` | Applied the *already-fitted* scaler to test data — fitting only on train data avoided leaking test-set statistics into preprocessing |
| `LogisticRegression(max_iter=1000)` | `sklearn.linear_model` | Baseline interpretable model |
| `RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10)` | `sklearn.ensemble` | Tree ensemble baseline; depth/leaf constraints limited overfitting |
| `XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)` | `xgboost` | Gradient boosting model; ended up as the best performer |
| `model.predict(X_test)` | — | Hard class predictions (0/1) at the default 0.5 threshold |
| `model.predict_proba(X_test)[:, 1]` | — | Predicted probabilities for the positive (churned) class — needed for ROC-AUC and custom thresholding |
| `classification_report(y_test, y_pred)` | `sklearn.metrics` | Printed precision/recall/F1 per class |
| `confusion_matrix(y_test, y_pred)` | `sklearn.metrics` | Returned the TN/FP/FN/TP counts as a 2x2 array |
| `roc_auc_score(y_test, y_pred_proba)` | `sklearn.metrics` | Computed the ROC-AUC — used probabilities, not hard predictions |
| `roc_curve(y_test, y_pred_proba)` | `sklearn.metrics` | Returned false-positive-rate/true-positive-rate pairs across all thresholds, for plotting |
| `precision_recall_curve(y_test, y_pred_proba)` | `sklearn.metrics` | Returned precision/recall at every possible threshold — used to pick a custom threshold (0.35) that favored recall |
| `(y_pred_proba >= custom_threshold).astype(int)` | — | Applied a manually chosen threshold instead of the sklearn default of 0.5 |
| `model.coef_[0]` | — | Logistic Regression's learned coefficients — used for feature-level interpretation |
| `model.feature_importances_` | — | XGBoost/Random Forest's built-in feature importance scores (based on split gain) |
| `joblib.dump(obj, path)` / `joblib.load(path)` | `joblib` | Saved/loaded the trained model and the feature column list to/from a `.pkl` file, so the model could be reused without retraining |

**Key decision:** picked threshold 0.35 (not the default 0.5) because the business case (cheap retention campaign) meant missing a real churner (false negative) was costlier than a false alarm — deliberately traded some precision for higher recall.

---

## Stage 6: Business Impact / ROI Analysis

**Goal:** Translate model output into a £ value, and compare against a naive baseline.

| Function | What it did |
|---|---|
| `confusion_matrix(...).ravel()` | Unpacked the 2x2 confusion matrix into four separate scalars: `tn, fp, fn, tp = cm.ravel()` |
| Manual formula (not a library function) | `net_value = (TP × success_rate × avg_order_value) − ((TP + FP) × campaign_cost)` — computed expected revenue recovered minus campaign spend |
| `np.arange(start, stop, step)` | Generated a range of campaign-cost values to test sensitivity (the break-even analysis) |
| Loop + list comprehension pattern | Recomputed net value for the model vs. a "blanket campaign" baseline across many cost assumptions, to find the crossover ("break-even") point |
| `plt.fill_between(x, y, 0, where=condition)` | Shaded the chart green/red depending on whether the model was winning or losing at each cost point |

**Key decision:** didn't stop at "the model predicts churn well" — explicitly built a comparison against the naive "contact everyone" baseline, since a model is only useful if it beats the simplest possible strategy. Found the model only wins above ~£6.50/customer campaign cost.

---

## Stage 7: Deployment

**Goal:** Build an interactive dashboard so non-technical users can explore results.

| Function/Concept | Library | What it did |
|---|---|---|
| `st.set_page_config(page_title=..., layout="wide")` | `streamlit` | Set the browser tab title and page width |
| `@st.cache_data` | `streamlit` | Cached the data-loading function so it doesn't reload the parquet file on every user interaction (performance optimization) |
| `st.tabs([...])` | `streamlit` | Created a tabbed interface (Risk Explorer / ROI Simulator) |
| `st.slider(label, min, max, default, step)` | `streamlit` | Interactive slider widgets for filtering/simulating (churn probability threshold, campaign cost, success rate) |
| `st.metric(label, value, delta=...)` | `streamlit` | Displayed a labeled number with optional up/down indicator, used for summary stats |
| `st.dataframe(df.style.format({...}))` | `streamlit` + `pandas.Styler` | Displayed a table with custom number formatting (e.g., percentages, currency) applied per column |
| `st.pyplot(fig)` | `streamlit` | Rendered a matplotlib figure inside the Streamlit app |
| `sys.path.append(str(Path(__file__).resolve().parents[1]))` | `sys`, `pathlib` | Added the project root to Python's import search path, so `from src.config import ...` worked when running the app from a different working directory |

**Key decision:** the app reads a *pre-computed* predictions file (`customer_predictions.parquet`) rather than loading raw data and retraining on every run — much faster, and avoided needing Kaggle credentials in the public cloud deployment.

**Deployment note:** had to adjust `.gitignore` to allow the specific processed predictions file and model `.pkl` to be committed (using a `!filename` override after a broader exclusion rule), since Streamlit Community Cloud deploys directly from GitHub and can't see gitignored files.

---

## Summary of Libraries Used

- **pandas** — data manipulation, joins, groupby/agg, datetime handling
- **numpy** — log transforms, numeric ranges
- **matplotlib / seaborn** — static visualizations
- **scikit-learn** — train/test split, scaling, Logistic Regression, Random Forest, all evaluation metrics
- **xgboost** — final chosen model
- **joblib** — model persistence
- **streamlit** — interactive dashboard
- **kagglehub / python-dotenv** — data ingestion and credential management

## Biggest Lessons Learned

1. **Validate the target variable before building anything else** — the Olist dataset's 97% one-time-buyer rate would have made the whole project meaningless; caught this early by checking `customer_unique_id` counts in EDA.
2. **Leakage can be subtle** — it's not always "the label column itself in the features." A feature that's *definitionally* almost the same as the label (recent activity vs. a recency-based label) leaks just as badly, and it's easy to miss if the resulting AUC just looks "good" instead of suspiciously perfect.
3. **A good model isn't automatically a useful one** — the break-even analysis showed the model only beats a naive baseline above a certain cost threshold. Always compare against the simplest possible alternative before claiming a model adds value.
