# Credit Risk: Probability of Default (PD) Model

## Purpose

Home Credit, a lending institution, needs to predict whether a loan applicant will default before approving a loan — approving a risky applicant loses money, rejecting a safe one loses business. This project builds a Probability of Default (PD) model using the real-world **Home Credit Default Risk** dataset (Kaggle, ~307,500 applicants, 122 raw features).

The goal wasn't just to produce a number — it was to build something a risk analyst could actually explain and defend: why features were dropped, why a threshold was chosen, why one model beat another, and why the predictions look the way they do.

**Note:** This project scopes to PD only. LGD (Loss Given Default) and EAD (Exposure at Default) were intentionally excluded — the public dataset has no reliable labels for either, and simulating them with random distributions would produce fabricated, non-defensible numbers.

---

## Dataset

- **Source:** Home Credit Default Risk (Kaggle)
- **File used:** `application_train.csv`
- **Rows:** 307,511 applicants
- **Target:** `TARGET` — 1 = payment difficulty/default, 0 = repaid normally
- **Class balance:** ~92% no default, ~8% default (imbalanced)

---

## Process & Problems Solved

### 1. Data Exploration
Profiled all 122 columns for missing data. Found 67 columns with some missing values, ranging from 0.0003% to ~70%.

### 2. Cleaning
- Dropped 49 columns with >40% missing values (mostly apartment/building detail columns like `COMMONAREA_AVG`).
- Filled remaining numeric gaps with **median** (not mean — income and similar columns are right-skewed, median resists outlier distortion).
- Filled categorical gaps with **mode**.

**Problem found:** `DAYS_EMPLOYED` contained a placeholder value of `365243` (over 1,000 years) representing unemployed/retired applicants — not a real number, and not caught by standard missing-value checks since it wasn't `NaN`. Fixed by flagging these rows in a new `NOT_EMPLOYED` column, then converting the placeholder to `NaN` and imputing properly.

### 3. Feature Engineering
Converted raw day-counts into interpretable features and built domain-driven ratios:
- `AGE_YEARS`, `EMPLOYED_YEARS` (from negative day-counts)
- `DEBT_TO_INCOME` = loan amount ÷ annual income
- `INCOME_PER_PERSON` = income ÷ family size
- `CREDIT_TO_ANNUITY` = loan amount ÷ periodic payment (rough loan-length proxy)

### 4. Encoding & Split
One-hot encoded all categorical columns (`pd.get_dummies`, `drop_first=True` to avoid redundant columns). Split 80/20 train/test, **stratified** on `TARGET` to preserve the 92/8 class ratio in both sets.

### 5. Modeling
Two models trained and compared:

| Model | Preprocessing | AUC-ROC |
|---|---|---|
| Logistic Regression | StandardScaler applied | 0.7455 |
| Random Forest (tuned: 200 trees, max_depth=15) | None needed | 0.7412 |

**Problem found:** Logistic Regression initially scored only 0.63 AUC — far below expected. Root cause: features were on wildly different scales (income in lakhs vs. ratios near 1), which distorts linear models. Applying `StandardScaler` brought it to 0.7455.

**Key finding:** Random Forest, despite being the more complex model, did not outperform scaled Logistic Regression. Feature importance analysis showed `EXT_SOURCE_2` and `EXT_SOURCE_3` (external bureau credit scores) dominate prediction power — these are pre-processed risk summaries from outside data, and their strength likely sets a ceiling that model complexity alone can't push past.

### 6. Threshold Selection
Default 0.5 threshold caught **zero** actual defaulters — a direct consequence of class imbalance compressing predicted probabilities well below 0.5 for nearly everyone. Checked the actual probability distribution and set the threshold at **0.15** (roughly matching the real 8% default rate), which caught ~30% of actual defaulters with a manageable false-positive rate.

### 7. Explainability (SHAP)
Used SHAP TreeExplainer to move beyond a black-box score. Confirms `EXT_SOURCE_2`/`EXT_SOURCE_3` as dominant drivers, with direction: low bureau scores push predicted risk up, high scores push it down — consistent with real-world credit logic. Engineered features (`CREDIT_TO_ANNUITY`, `DEBT_TO_INCOME`, `EMPLOYED_YEARS`) also show up meaningfully in the ranking.

---

## Models Used
- **Logistic Regression** (scikit-learn) — linear baseline, scaled inputs
- **Random Forest Classifier** (scikit-learn) — ensemble of 200 decision trees
- **SHAP TreeExplainer** — model interpretability layer

---

## Key Learnings

- **Missing ≠ obviously missing.** The `365243` placeholder in `DAYS_EMPLOYED` looked like valid data and passed a standard `isnull()` check — real-world data hides errors as plausible-looking numbers, not just blanks.
- **Model complexity isn't automatically better.** Random Forest didn't beat a properly-scaled Logistic Regression here — the ceiling was set by feature quality (`EXT_SOURCE` scores), not model choice.
- **Accuracy is misleading on imbalanced data.** With ~92% non-defaulters, a model predicting "safe" for everyone would show 92% accuracy while catching zero real risk — AUC-ROC was used instead since it measures ranking quality, not raw correctness.
- **Default thresholds (0.5) don't work on imbalanced problems.** Had to derive a threshold from the actual probability distribution rather than assuming the textbook default.
- **Preprocessing decisions are trade-offs, not one-time fixes.** Dropping `EXT_SOURCE_1` (>40% missing, per a consistent rule) likely cost some model performance once `EXT_SOURCE_2/3` proved dominant — a legitimate, evidence-based decision made without hindsight, worth revisiting with a relaxed threshold as future work.

---

## Future Improvements
- Test a relaxed missing-value threshold to see if retaining `EXT_SOURCE_1` (currently dropped at >40% missing) improves AUC.
- Hyperparameter tuning via GridSearch/RandomizedSearch rather than manual adjustment.
- Cost-based threshold selection using actual false-negative/false-positive cost estimates instead of the default-rate heuristic used here.
- Convert exploratory notebook into modular `.py` scripts for production-style structure.

---

## Tools
Python, pandas, NumPy, scikit-learn, SHAP, matplotlib