# ML Project 1 – Predicting coronary heart disease (BRFSS 2015)

EPFL CS-433 Machine Learning, Fall 2026 – Class Project 1.

We predict whether a respondent of the 2015 Behavioral Risk Factor Surveillance System
(BRFSS) has coronary heart disease (`_MICHD`, label in {-1, 1}) from 321 self-reported
survey answers. Everything is implemented with **Python and NumPy only** (no pandas,
no scikit-learn), as required by the project rules.

**Final model:** logistic regression trained with gradient descent on 256 engineered
features, with a decision threshold tuned for F1.
**5-fold cross-validation:** F1 = 0.4275 ± 0.0019 (precision 0.348, recall 0.553, accuracy 0.869).

The full description of the method and of the experiments is in the report:
[`report/report.pdf`](report/report.pdf).

---

## 1. Quick start: reproduce the submission

**Requirements:** Python ≥ 3.9 and NumPy. Matplotlib is used only in the notebooks.

1. Download `x_train.csv`, `y_train.csv` and `x_test.csv` from the
   [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
   and put them, with their original names, in the folder `data/`
   (this folder is not tracked by git):

   ```
   data/
   ├── x_train.csv
   ├── y_train.csv
   └── x_test.csv
   ```

2. From the root of the repository, run:

   ```bash
   python run.py
   ```

3. The predictions are written to **`submission.csv`** (columns `Id, Prediction`,
   with `Prediction` in {-1, 1}). This is exactly the file we submitted to AIcrowd.

The first run reads the CSV files (a few minutes) and saves them as `.npy` files in
`cache/`; the following runs load the cache and take about one minute in total.
The script is fully deterministic: it trains the model from scratch every time, with
no random initialization and no sampling, so it always produces the same file.

Expected output:

```
1. Loading the data
   train (328135, 321), test (109379, 321)
2. Cleaning
   95 columns, NaN: train 5.2%, test 5.2%
3. Preprocessing (fitted on the training set)
   feature matrix: train (328135, 256), test (109379, 256)
4. Logistic regression (1000 GD steps, gamma = 16/L)
   L = 8.286, training loss = 0.22099
5. Decision threshold (maximum F1 on the training set)
   threshold = 0.1948, training F1 = 0.4307
6. Predictions on the test set
   predicted positives: 14.0%  -> submission.csv
```

---

## 2. Repository structure

| File / folder | Content |
|---|---|
| `run.py` | Complete pipeline from the raw CSV files to `submission.csv` (cleaning → features → model → threshold → predictions). |
| `implementations.py` | The 6 required methods (Table 1 of the project description) and their helpers (losses, gradients, sigmoid). |
| `cleaning.py` | Cleaning rules: special codes, structural zeros, removal of redundant columns, consistency checks, impossible values, exact collinearities. |
| `preprocessing.py` | Feature processing with separate `fit` / `transform` steps: imputation, missing indicators, log transform, standardization, one-hot encoding, removal of duplicated columns. |
| `helpers.py` | Course helpers: `load_csv_data`, `create_csv_submission`. |
| `notebooks/notebookmatteo.ipynb` | Step-by-step construction of the cleaning pipeline (321 → 95 columns), with every check and its output. |
| `notebooks/Giovanni.ipynb` | Analysis of the columns: missing values, informativeness, redundant columns. |
| `notebooks/notebookMario.ipynb` | Stratified cross-validation, evaluation metrics, threshold selection, comparison of the 6 methods, step-size tuning, interaction features. |
| `notebooks/stats2.ipynb`, `pulizia_dataset.ipynb` | First exploratory analysis and first version of the cleaning. |
| `CUTTING_COLONNE.md` | Column-by-column log of every cleaning decision and its motivation. |
| `report/` | LaTeX source and PDF of the 2-page report. |

The notebooks and `CUTTING_COLONNE.md` are written in Italian. The code and its
documentation are in English, except for some comments in `cleaning.py` and
`preprocessing.py`.

---

## 3. Pipeline

All steps that use statistics of the data (special codes, NaN fractions, medians,
means, standard deviations, one-hot levels, consistency checks) are computed **on the
training set only** and applied unchanged to the test set. During cross-validation,
the feature processing is refitted **inside each fold**.

### 3.1 Data cleaning (`run.py: clean_data`, `cleaning.py`) – 321 → 95 variables

1. **Columns removed a priori:** identifiers and dates, call-management variables,
   survey weights, variables of the landline / cell-phone versions of the questionnaire
   (their NaN only reveals the phone type), and `HAREHAB1` (cardiac rehabilitation,
   only filled in by positive respondents: label leakage). Raw columns that have a clean
   CDC-computed version (`WEIGHT2`, `HEIGHT3`, …) are also removed.
2. **Special codes** (BRFSS codebook):
   - structural zeros: when a parent question says "no", the unasked quantity is set to 0
     (e.g. non-drinkers drink 0 drinks; non-diabetics have 0 years with diabetes);
   - "none" codes (88, 888, 555) → 0;
   - "don't know / refused" codes (7/77/777, 9/99/999, …) → NaN. The codes are detected
     on the training set (`find_special_codes`).
3. **Sparse columns:** columns with more than 60% NaN in the training set are dropped
   (mostly state-optional modules). Insulin use is kept and merged with the diabetes
   status into `INSULIN_CAT`.
4. **Redundant columns:** columns that are an exact function of a column we keep are
   removed (purity test: for every value of A, B always takes the same value).
5. **Groups 1–5 of `cleaning.py`:**
   1. raw answers with mixed units in the same number (e.g. `1xx` = per week,
      `2xx` = per month) are replaced by their CDC-computed version; `IMFVPLAC` becomes
      the binary `IMFV_LAVORO`; `EXRACT11/21` and `_HCVU651` are dropped;
   2. structural NaNs of skip-logic questions are set to the implied answer
      (e.g. not hypertensive → no blood-pressure medication);
   3. duplicated categorical columns are removed;
   4. columns that are exact sums of others are removed; physically impossible values
      (e.g. more than 10,080 minutes of activity per week) become NaN;
   5. exact collinearities (insulin, arthritis limitations, smoking, asthma) are fixed.

### 3.2 Feature generation (`preprocessing.py`) – 95 variables → 256 features

- **Bias** column of ones.
- **34 continuous variables:** NaN replaced by the median of the training set, plus one
  binary missing indicator for each group of columns that are missing in exactly the
  same rows; `log(1 + x)` on 21 long-tailed variables (alcohol, fruit and vegetables,
  physical activity); z-score standardization.
- **61 categorical variables:** one-hot encoding, with NaN treated as its own category;
  the most frequent level is the reference and is left out.
- Constant columns and identical columns are removed automatically.

The resulting matrix has no NaN and full rank (condition number of XᵀX ≈ 7·10⁴), so
that `least_squares` can be solved with the normal equations.

### 3.3 Model and decision threshold (`run.py`)

- Labels are converted from {-1, 1} to {0, 1}.
- `logistic_regression` with 1000 gradient-descent steps, starting from w = 0, with
  step size γ = 16 / L, where L is the largest eigenvalue of XᵀX / N (power method,
  L ≈ 8.3). γ = 4 / L is guaranteed to converge but is too slow; γ = 64 / L diverges.
- Classes are imbalanced (8.8% positives), so the threshold 0.5 is not used: the
  threshold on the predicted probability is the one that maximizes F1 **on the training
  set** (searched among 200 percentiles of the scores). The final threshold is 0.195.

### 3.4 Cross-validation (`notebooks/notebookMario.ipynb`)

- **Stratified 5-fold cross-validation** (each fold has the same 8.83% of positives),
  seed 1. The same folds are used for every experiment.
- For each fold: preprocessing fitted on the training part, model trained on the
  training part, threshold chosen on the training part, metrics computed on the
  validation part.
- We report the mean ± standard deviation over the folds of F1, precision, recall and
  accuracy.
- Two variants are compared fold by fold on the same folds: a difference is considered
  real only if the mean of the per-fold differences is larger than their standard
  deviation.

---

## 4. Results (5-fold cross-validation)

| Method | F1 (validation) |
|---|---|
| least squares | 0.4201 ± 0.0023 |
| ridge regression (λ = 10⁻⁶) | 0.4201 ± 0.0023 |
| GD on MSE (200 steps, γ = 1/L) | 0.4165 ± 0.0019 |
| SGD on MSE (one epoch, batch size 1) | 0.4053 ± 0.0050 |
| logistic regression (200 steps, γ = 4/L) | 0.4000 ± 0.0028 |
| regularized logistic regression (200 steps, γ = 4/L) | 0.4005 ± 0.0028 |
| **logistic regression (1000 steps, γ = 16/L) – final model** | **0.4275 ± 0.0019** |

Reference points: always predicting "healthy" gives accuracy 91.2% but F1 = 0; always
predicting "sick" gives F1 = 0.162. The ablation study (threshold, cleaning, encoding
choices, interaction features) is reported in Table II of the report.

---

## 5. The six required methods (`implementations.py`)

| Function | Method |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression with gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression with stochastic gradient descent (batch size 1) |
| `least_squares(y, tx)` | Least squares with the normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression with the normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression with gradient descent, y ∈ {0, 1} |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression (penalty λ‖w‖²) with gradient descent, y ∈ {0, 1} |

All functions return `(w, loss)`: the last weight vector and the loss computed on the
whole dataset (MSE with the factor 1/2, or the mean negative log-likelihood). For the
regularized methods the returned loss does not include the penalty term. All vectors
are 1D arrays.

---

## Team

Mario [SURNAME], Giovanni Pavan, Matteo Barbone
