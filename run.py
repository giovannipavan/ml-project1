"""Reproduce our final AIcrowd submission for Project 1 (BRFSS 2015, target _MICHD).

Usage (from the repository root):

    python run.py

Input : data/x_train.csv, data/y_train.csv, data/x_test.csv  (names unchanged)
Output: submission.csv  (columns Id, Prediction, with Prediction in {-1, 1})

The script runs the whole pipeline from the raw CSV files, with NumPy only:

  1. Load the data (from the .npy cache in cache/ if it exists, else from the CSVs).
  2. Cleaning, with fixed rules (cleaning.py):
       a. drop identifiers, call-management columns, survey weights, landline /
          cell-phone questionnaire columns, the leaking HAREHAB1, raw duplicates;
       b. structural zeros, "none" codes -> 0, "don't know / refused" codes -> NaN
          (the codes are found on the TRAINING set and applied unchanged to the test set);
       c. drop columns with more than 60% NaN (on the training set), build INSULIN_CAT;
       d. drop columns that are exact functions of a column we keep;
       e. groups 1-5 of cleaning.py (mixed-coding raw columns, IMFVPLAC, _HCVU651,
          structural NaNs, duplicated categoricals, exact sums, impossible values,
          exact collinearities).
     Every decision that depends on the data is taken on the training set only.
     Result: 95 columns, with NaNs.
  3. Preprocessing (preprocessing.py), fitted on the training set only:
       bias | continuous (median imputation, log(1+x) on long tails, z-score)
            | missing-value indicators | one-hot of the categoricals (NaN = own level)
     Result: 256 columns, no NaNs.
  4. Model: logistic regression with gradient descent (implementations.py),
     1000 steps, step size gamma = 16 / L, where L is the largest eigenvalue of
     X^T X / N (power method). Target y in {0, 1}.
  5. Decision threshold on the predicted probability: the one that maximizes F1 on
     the training set (classes are imbalanced: 8.8% positives).
  6. Predictions on the test set, mapped to {-1, 1}, written with create_csv_submission.

5-fold stratified cross-validation of this exact configuration (notebooks/notebookMario.ipynb):
F1 = 0.4275 +- 0.0019, accuracy = 0.869.

The whole run is deterministic (no random initialization, no sampling).
"""

import time
from pathlib import Path

import numpy as np

from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression, sigmoid
from preprocessing import fit_transform_prep, transform_prep
from cleaning import (
    read_header, drop_columns,
    fill_structural_zeros, recode_none_to_zero, find_special_codes, special_codes_to_nan,
    rimuovi_grezze, trasforma_imfv, rimuovi_hcvu651,
    controlli_gruppo2, trasforma_gruppo2, rimuovi_gruppo3, regole_fisse_gruppo4,
    NO_ARTRITE,
)

# ---------------------------------------------------------------------------
# Paths and hyperparameters (chosen with 5-fold cross-validation)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data"
CACHE_PATH = ROOT / "cache"
OUTPUT = ROOT / "submission.csv"

MAX_ITERS = 1000        # gradient-descent steps of the logistic regression
STEP_FACTOR = 16        # gamma = STEP_FACTOR / L   (4 is too slow, 64 diverges)
NAN_THRESHOLD = 0.6     # drop columns with more than 60% NaN in the training set

# Candidate columns kept apart for the ablation study; the final model does NOT use them.
CANDIDATES = ["INSULIN", "CVDASPRN", "DRADVISE"]

# Columns removed a priori (see CUTTING_COLONNE.md and notebooks/notebookmatteo.ipynb).
DROP_A_PRIORI = [
    # identifiers and dates
    "Id", "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "STATERES",
    # call management
    "DISPCODE", "SEQNO", "_PSU", "CTELENUM", "PVTRESD1", "CELLFON3",
    "NUMHHOL2", "CPDEMO1", "QSTVER", "QSTLANG", "MSCODE",
    # survey weights
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE", "_DUALCOR", "_LLCPWT",
    # landline / cell-phone version of the questionnaire (the NaN only tells the phone type)
    "COLGHOUS", "LADULT", "NUMADULT", "NUMMEN", "NUMWOMEN", "NUMPHON2",
    "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2", "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT",
    # leakage: only filled in by people who had a heart attack
    "HAREHAB1",
    # raw columns with a clean CDC version, and dates
    "WEIGHT2", "HEIGHT3", "EXEROFT1", "EXERHMM1", "EXEROFT2", "EXERHMM2", "FLSHTMY2", "HIVTSTD3",
]

# Columns that are an exact (deterministic) function of a column we keep.
DROP_COPIES = ["_RFHLTH", "_RFHYPE5", "_RFCHOL", "_LTASTH1", "_CASTHM1", "_DRDXAR1",
               "_EDUCAG", "_INCOMG", "_RFSMOK3", "SMOKDAY2", "_RFSEAT2", "_RFSEAT3",
               "_AIDTST3", "_RACEG21", "_RACEGR3", "_RACE_G1", "_AGE_G", "_TOTINDA",
               "_PA30021", "_PAINDX1", "_PASTAE1", "HTIN4", "_RFBMI5", "FC60_"]


# ---------------------------------------------------------------------------
# Step 1: data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load the raw data, from the .npy cache if present (it is the output of load_csv_data)."""
    names = ["x_train", "x_test", "y_train", "train_ids", "test_ids"]
    if all((CACHE_PATH / f"{n}.npy").exists() for n in names):
        print("   loading from the .npy cache")
        return tuple(np.load(CACHE_PATH / f"{n}.npy") for n in names)
    print("   reading the CSV files (a few minutes) and saving the .npy cache")
    data = load_csv_data(str(DATA_PATH))
    CACHE_PATH.mkdir(exist_ok=True)
    for n, a in zip(names, data):
        np.save(CACHE_PATH / f"{n}.npy", a)
    return data


# ---------------------------------------------------------------------------
# Step 2: cleaning (fixed rules; data-dependent choices made on the training set)
# ---------------------------------------------------------------------------
def insulin_category(diabetes, insulin):
    """0 not diabetic, 1 diabetic without the module, 2 unknown,
    3 diabetic using insulin, 4 diabetic not using insulin."""
    cat = np.zeros(len(diabetes), dtype=int)
    cat[np.isnan(diabetes)] = 2
    cat[diabetes == 1] = 1
    cat[(diabetes == 1) & (insulin == 1)] = 3
    cat[(diabetes == 1) & (insulin == 2)] = 4
    return cat


def drop_both(x_tr, x_te, cols, names):
    """Drop the same columns from train and test, keeping them aligned."""
    x_tr, cols_tr = drop_columns(x_tr, cols, names)
    x_te, cols_te = drop_columns(x_te, cols, names)
    assert cols_tr == cols_te, "train and test are no longer aligned"
    return x_tr, x_te, cols_tr


def check_collinearities(X, cols):
    """Checks of the exact collinearities D, E, F, G, computed on the TRAINING set."""
    j = {c: i for i, c in enumerate(cols)}
    col = lambda n: X[:, j[n]]
    equal = lambda a, b: bool(np.array_equal(a, b))
    ok = {"D": False, "E": "SMOKE100" in j, "F": {}, "G": False}
    if "INSULIN_CAT" in j and len(np.unique(col("INSULIN_CAT"))) == 5:
        ins, dia = col("INSULIN_CAT"), col("DIABETE3")
        ok["D"] = equal(ins == 2, np.isnan(dia)) and equal(np.isin(ins, [1, 3, 4]), dia == 1)
    for c, code in NO_ARTRITE.items():
        if c in j and "HAVARTH3" in j:
            ok["F"][c] = equal(col(c) == code, col("HAVARTH3") == 2)
    if "ASTHMA3" in j and "_ASTHMS1" in j:
        ok["G"] = equal(col("_ASTHMS1") == 3, col("ASTHMA3") == 2)
    return ok


def fix_collinearities(X, cols, ok):
    """Apply the corrections D, E, F, G (decided on the training set) to any matrix.
      D. INSULIN_CAT: 0, 1, 2 -> 0 (no evidence of insulin use; diabetes is in DIABETE3)
      E. SMOKE100 dropped (already contained in _SMOKER3)
      F. arthritis limitations: "no arthritis" -> "not limited"
      G. ASTHMA3 dropped (_ASTHMS1 contains it)"""
    X = X.copy()
    j = {c: i for i, c in enumerate(cols)}
    if ok["D"]:
        X[np.isin(X[:, j["INSULIN_CAT"]], [0, 1, 2]), j["INSULIN_CAT"]] = 0
    for c, passed in ok["F"].items():
        if passed:
            code = NO_ARTRITE[c]
            X[X[:, j[c]] == code, j[c]] = code - 1
    to_drop = (["SMOKE100"] if ok["E"] else []) + (["ASTHMA3"] if ok["G"] else [])
    return drop_columns(X, cols, to_drop)


def clean_data(x_tr, x_te):
    """From the 321 raw columns to the 95 clean columns (still with NaNs)."""
    cols = read_header(DATA_PATH / "x_train.csv")
    if len(cols) == x_tr.shape[1] + 1:          # load_csv_data already removes 'Id'
        cols = cols[1:]
    assert len(cols) == x_tr.shape[1] == x_te.shape[1]

    # a. columns removed a priori
    x_tr, x_te, cols = drop_both(x_tr, x_te, cols, DROP_A_PRIORI)

    # b. special codes (order matters: steps 1-2 read the original codes, step 3 deletes them)
    x_tr = recode_none_to_zero(fill_structural_zeros(x_tr, cols), cols)
    x_te = recode_none_to_zero(fill_structural_zeros(x_te, cols), cols)
    codes = find_special_codes(x_tr, cols)                       # found on the train only
    x_tr = special_codes_to_nan(x_tr, cols, codes)
    x_te = special_codes_to_nan(x_te, cols, codes)

    # c. columns with too many NaNs (fraction computed on the train) + INSULIN_CAT
    nan_frac = np.isnan(x_tr).mean(axis=0)
    too_many_nan = [c for c, f in zip(cols, nan_frac) if f > NAN_THRESHOLD and c not in CANDIDATES]
    ix = {c: i for i, c in enumerate(cols)}
    ins_tr = insulin_category(x_tr[:, ix["DIABETE3"]], x_tr[:, ix["INSULIN"]])
    ins_te = insulin_category(x_te[:, ix["DIABETE3"]], x_te[:, ix["INSULIN"]])
    x_tr, x_te, cols = drop_both(x_tr, x_te, cols, too_many_nan + ["INSULIN"])
    x_tr, x_te = np.column_stack([x_tr, ins_tr]), np.column_stack([x_te, ins_te])
    cols = cols + ["INSULIN_CAT"]

    # d. exact copies, and the candidate columns not used by the final model
    x_tr, x_te, cols = drop_both(x_tr, x_te, cols, DROP_COPIES + CANDIDATES)

    # e. groups 1-5: the checks are run on the train, the same columns are dropped from the test
    x_tr, cols_new, _ = rimuovi_grezze(x_tr, cols)                       # 1a
    x_te, _ = drop_columns(x_te, cols, [c for c in cols if c not in cols_new])
    cols = cols_new

    x_tr, cols_new = trasforma_imfv(x_tr, cols)                          # 1b
    x_te, cols_te = trasforma_imfv(x_te, cols)
    assert cols_new == cols_te
    cols = cols_new

    x_tr, cols_new, _ = rimuovi_hcvu651(x_tr, cols)                      # 1c
    x_te, _ = drop_columns(x_te, cols, [c for c in cols if c not in cols_new])
    cols = cols_new

    ok2 = controlli_gruppo2(x_tr, cols)                                  # 2
    x_tr, cols_new = trasforma_gruppo2(x_tr, cols, ok2)
    x_te, cols_te = trasforma_gruppo2(x_te, cols, ok2)
    assert cols_new == cols_te
    cols = cols_new

    x_tr, cols_new = rimuovi_gruppo3(x_tr, cols)                         # 3
    x_te, cols_te = rimuovi_gruppo3(x_te, cols)
    assert cols_new == cols_te
    cols = cols_new

    x_tr, cols_new = regole_fisse_gruppo4(x_tr, cols)                    # 4
    x_te, cols_te = regole_fisse_gruppo4(x_te, cols)
    assert cols_new == cols_te
    cols = cols_new

    ok5 = check_collinearities(x_tr, cols)                               # 5
    x_tr, cols_new = fix_collinearities(x_tr, cols, ok5)
    x_te, cols_te = fix_collinearities(x_te, cols, ok5)
    assert cols_new == cols_te
    return x_tr, x_te, cols_new


# ---------------------------------------------------------------------------
# Steps 4-5: model and decision threshold
# ---------------------------------------------------------------------------
def lipschitz_constant(X, iters=30, seed=0):
    """Largest eigenvalue of X^T X / N (power method), to set the GD step size."""
    v = np.random.default_rng(seed).standard_normal(X.shape[1])
    lam = 1.0
    for _ in range(iters):
        v = X.T @ (X @ v) / X.shape[0]
        lam = np.linalg.norm(v)
        v /= lam
    return lam


def f1_score(y_true, y_pred):
    """F1 of the positive class (value 1)."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y, scores, percentiles=np.linspace(50, 99.5, 200)):
    """Threshold on the scores that maximizes F1, searched among percentiles of the scores."""
    candidates = np.unique(np.percentile(scores, percentiles))
    f1s = [f1_score(y, (scores >= t).astype(int)) for t in candidates]
    k = int(np.argmax(f1s))
    return candidates[k], f1s[k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()

    print("1. Loading the data")
    x_train, x_test, y_train, train_ids, test_ids = load_data()
    print(f"   train {x_train.shape}, test {x_test.shape}")

    print("2. Cleaning")
    x_tr, x_te, cols = clean_data(x_train, x_test)
    del x_train, x_test
    print(f"   {len(cols)} columns, NaN: train {100 * np.isnan(x_tr).mean():.1f}%, "
          f"test {100 * np.isnan(x_te).mean():.1f}%")

    print("3. Preprocessing (fitted on the training set)")
    p, X_tr = fit_transform_prep(x_tr, cols)
    X_te = transform_prep(x_te, p)
    X_tr, X_te = X_tr.astype(np.float64), X_te.astype(np.float64)
    print(f"   feature matrix: train {X_tr.shape}, test {X_te.shape}")

    print(f"4. Logistic regression ({MAX_ITERS} GD steps, gamma = {STEP_FACTOR}/L)")
    y01 = (y_train == 1).astype(np.float64)                  # labels in {0, 1}
    L = lipschitz_constant(X_tr)
    t0 = time.perf_counter()
    w, loss = logistic_regression(y01, X_tr, np.zeros(X_tr.shape[1]), MAX_ITERS, STEP_FACTOR / L)
    print(f"   L = {L:.3f}, training loss = {loss:.5f}, time {time.perf_counter() - t0:.0f}s")

    print("5. Decision threshold (maximum F1 on the training set)")
    threshold, f1_train = best_threshold(y01, sigmoid(X_tr @ w))
    print(f"   threshold = {threshold:.4f}, training F1 = {f1_train:.4f}")

    print("6. Predictions on the test set")
    y_pred = np.where(sigmoid(X_te @ w) >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, str(OUTPUT))
    print(f"   predicted positives: {100 * (y_pred == 1).mean():.1f}%  -> {OUTPUT.name}")
    print(f"done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
