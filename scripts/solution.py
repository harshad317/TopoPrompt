"""
Kaggle Playground 2026 - Irrigation Need Prediction
Metric: Balanced Accuracy (multiclass: Low / Medium / High)
High class is only ~3.3% → heavy imbalance, must use class weights.

Run with:
    uv run scripts/solution.py
"""

import os, sys

# Fix libomp path on macOS without Homebrew (uses sklearn's bundled libomp)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_libomp_dir = os.path.join(
    _project_root, ".venv", "lib", "python3.13", "site-packages", "sklearn", ".dylibs"
)
if os.path.isdir(_libomp_dir) and "DYLD_LIBRARY_PATH" not in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = _libomp_dir
    os.execv(sys.executable, [sys.executable] + sys.argv)  # restart with env set

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")

SEED = 45          # fixed validation split seed per evaluation protocol
N_FOLDS = 5
TARGET = "Irrigation_Need"
LABEL_ORDER = ["Low", "Medium", "High"]  # encoded as 0, 1, 2

# Paths
DATA_DIR = os.path.join(_project_root, "Data")
PRED_DIR = os.path.join(_project_root, "Predictions")
os.makedirs(PRED_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"Train: {train.shape}  Test: {test.shape}")
print("Target distribution:\n", train[TARGET].value_counts())

# ── 2. Encode target ──────────────────────────────────────────────────────────

label_enc = LabelEncoder()
label_enc.classes_ = np.array(LABEL_ORDER)
train["label"] = label_enc.transform(train[TARGET])

# ── 3. Feature engineering ────────────────────────────────────────────────────

CAT_COLS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
ENG_COLS = [
    "water_stress", "et_proxy", "effective_rain", "rain_per_area",
    "prev_irr_per_ha", "aridity", "heat_index", "soil_quality",
]

def add_features(df):
    df = df.copy()
    # High temp + low moisture + low rainfall → more irrigation needed
    df["water_stress"]    = df["Temperature_C"] / (df["Soil_Moisture"] + 1) / (df["Rainfall_mm"] + 1)
    # Evapotranspiration proxy
    df["et_proxy"]        = df["Temperature_C"] * df["Sunlight_Hours"] * (1 - df["Humidity"] / 100)
    # Effective rainfall after soil absorption
    df["effective_rain"]  = df["Rainfall_mm"] * (1 - df["Soil_Moisture"] / 100)
    # Rainfall per unit area
    df["rain_per_area"]   = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 0.01)
    # Previous irrigation relative to area
    df["prev_irr_per_ha"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 0.01)
    # Aridity index proxy
    df["aridity"]         = (df["Temperature_C"] * df["Wind_Speed_kmh"]) / (df["Rainfall_mm"] + 1)
    # Heat-humidity feel index
    df["heat_index"]      = df["Temperature_C"] * df["Humidity"] / 100
    # Soil quality (organic carbon vs salinity)
    df["soil_quality"]    = df["Organic_Carbon"] / (df["Electrical_Conductivity"] + 0.01)
    return df

train = add_features(train)
test  = add_features(test)

FEAT_COLS = NUM_COLS + ENG_COLS + CAT_COLS  # used by CatBoost

# Ordinal-encode categoricals for LGB / XGB
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_cat_enc = oe.fit_transform(train[CAT_COLS])
test_cat_enc  = oe.transform(test[CAT_COLS])

X_num  = train[NUM_COLS + ENG_COLS].values
X_test_num = test[NUM_COLS + ENG_COLS].values

X      = np.hstack([X_num, train_cat_enc])
X_test = np.hstack([X_test_num, test_cat_enc])
y      = train["label"].values

lgb_cat_indices = list(range(len(NUM_COLS + ENG_COLS), X.shape[1]))

# ── 4. Class weights ──────────────────────────────────────────────────────────

class_counts  = np.bincount(y)
class_weights = len(y) / (len(class_counts) * class_counts)
sample_weights = class_weights[y]
print(f"\nClass counts:  {dict(zip(LABEL_ORDER, class_counts))}")
print(f"Class weights: {dict(zip(LABEL_ORDER, class_weights.round(3)))}")

# ── 5. CV setup (seed=45, 5-fold ≈ 20% val each fold per protocol) ───────────

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
n_classes = 3

oof_lgb  = np.zeros((len(X), n_classes))
oof_xgb  = np.zeros((len(X), n_classes))
oof_cat  = np.zeros((len(X), n_classes))
pred_lgb = np.zeros((len(X_test), n_classes))
pred_xgb = np.zeros((len(X_test), n_classes))
pred_cat = np.zeros((len(X_test), n_classes))

best_iters_lgb = []
best_iters_xgb = []

# ── 6. LightGBM ───────────────────────────────────────────────────────────────

print("\n" + "─"*50)
print("LightGBM")
print("─"*50)

lgb_params = dict(
    objective        = "multiclass",
    num_class        = 3,
    metric           = "None",
    n_estimators     = 3000,
    learning_rate    = 0.05,
    num_leaves       = 127,
    min_child_samples= 20,
    subsample        = 0.8,
    subsample_freq   = 1,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    is_unbalance     = True,
    random_state     = SEED,
    n_jobs           = -1,
    verbose          = -1,
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        sample_weight    = sample_weights[tr_idx],
        eval_set         = [(X_val, y_val)],
        categorical_feature = lgb_cat_indices,
        callbacks        = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    oof_lgb[val_idx] = model.predict_proba(X_val)
    pred_lgb += model.predict_proba(X_test) / N_FOLDS
    best_iters_lgb.append(model.best_iteration_)

    ba = balanced_accuracy_score(y_val, np.argmax(oof_lgb[val_idx], axis=1))
    print(f"  Fold {fold+1}: BA = {ba:.4f}  (best iter: {model.best_iteration_})")

print(f"LGB OOF BA: {balanced_accuracy_score(y, np.argmax(oof_lgb, axis=1)):.4f}")

# ── 7. XGBoost ────────────────────────────────────────────────────────────────

print("\n" + "─"*50)
print("XGBoost")
print("─"*50)

xgb_params = dict(
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    n_estimators     = 3000,
    learning_rate    = 0.05,
    max_depth        = 7,
    min_child_weight = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0,
    tree_method      = "hist",
    random_state     = SEED,
    n_jobs           = -1,
    verbosity        = 0,
    early_stopping_rounds = 100,
)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        sample_weight = sample_weights[tr_idx],
        eval_set      = [(X_val, y_val)],
        verbose       = 500,
    )
    oof_xgb[val_idx] = model.predict_proba(X_val)
    pred_xgb += model.predict_proba(X_test) / N_FOLDS
    best_iters_xgb.append(model.best_iteration)

    ba = balanced_accuracy_score(y_val, np.argmax(oof_xgb[val_idx], axis=1))
    print(f"  Fold {fold+1}: BA = {ba:.4f}  (best iter: {model.best_iteration})")

print(f"XGB OOF BA: {balanced_accuracy_score(y, np.argmax(oof_xgb, axis=1)):.4f}")

# ── 8. CatBoost ───────────────────────────────────────────────────────────────

print("\n" + "─"*50)
print("CatBoost")
print("─"*50)

X_cb      = train[FEAT_COLS].values
X_test_cb = test[FEAT_COLS].values
cat_col_indices = [FEAT_COLS.index(c) for c in CAT_COLS]

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cb, y)):
    X_tr, X_val = X_cb[tr_idx], X_cb[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = CatBoostClassifier(
        iterations         = 3000,
        learning_rate      = 0.05,
        depth              = 7,
        l2_leaf_reg        = 3,
        loss_function      = "MultiClass",
        eval_metric        = "TotalF1",
        auto_class_weights = "Balanced",
        cat_features       = cat_col_indices,
        random_seed        = SEED,
        early_stopping_rounds = 100,
        verbose            = 500,
    )
    model.fit(
        X_tr, y_tr,
        sample_weight = sample_weights[tr_idx],
        eval_set      = (X_val, y_val),
    )
    oof_cat[val_idx] = model.predict_proba(X_val)
    pred_cat += model.predict_proba(X_test_cb) / N_FOLDS

    ba = balanced_accuracy_score(y_val, np.argmax(oof_cat[val_idx], axis=1))
    print(f"  Fold {fold+1}: BA = {ba:.4f}")

print(f"CAT OOF BA: {balanced_accuracy_score(y, np.argmax(oof_cat, axis=1)):.4f}")

# ── 9. Ensemble weight search ─────────────────────────────────────────────────

print("\n" + "─"*50)
print("Ensemble weight search")
print("─"*50)

best_ba, best_w = 0, (1/3, 1/3, 1/3)
for w1 in np.arange(0.1, 0.8, 0.05):
    for w2 in np.arange(0.1, 0.8, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0.05:
            continue
        blend = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cat
        ba = balanced_accuracy_score(y, np.argmax(blend, axis=1))
        if ba > best_ba:
            best_ba, best_w = ba, (w1, w2, w3)

print(f"Best weights — LGB:{best_w[0]:.2f}  XGB:{best_w[1]:.2f}  CAT:{best_w[2]:.2f}")
print(f"Ensemble OOF BA: {best_ba:.4f}")

# Summary lines (greppable by lead agent)
avg_best_iter = int(np.mean(best_iters_lgb + best_iters_xgb))
print(f"val_balanced_accuracy_score: {best_ba:.6f}")
print(f"best_iteration: {avg_best_iter}")

# ── 10. Generate submission ───────────────────────────────────────────────────

test_blend = best_w[0] * pred_lgb + best_w[1] * pred_xgb + best_w[2] * pred_cat
test_pred  = label_enc.inverse_transform(np.argmax(test_blend, axis=1))

sub["Irrigation_Need"] = test_pred
pred_path = os.path.join(PRED_DIR, "prediction_irr_need.csv")
sub.to_csv(pred_path, index=False)

print(f"\nPrediction saved → {pred_path}")
print(f"Prediction distribution:\n{pd.Series(test_pred).value_counts()}")
