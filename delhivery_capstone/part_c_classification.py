"""Part C: train the delay classifier, select features and choose an alert threshold."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.shipment_prep import FEATURES, NOISE_COLUMNS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

modelling = pd.read_csv(DATA / "modelling_table.csv")
X = modelling[FEATURES]
y = modelling["is_delayed"].astype(int)

# Two-stage stratified split: 840 train, 180 validation, 180 test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


def score_labels(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# Blind baseline: classify every parcel as on time (not delayed).
model_rows = [{"model": "blind_all_on_time", **score_labels(y_val, np.zeros(len(y_val), dtype=int))}]

# Logistic regression pipeline.
logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000)),
])
logistic.fit(X_train, y_train)
model_rows.append({"model": "logistic_regression", **score_labels(y_val, logistic.predict(X_val))})

# Select the decision-tree depth using validation F1; ties go to the smaller depth.
tree_depths = [2, 3, 4, 5, 6, 8, 10]
tree_results = []
for depth in tree_depths:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    val_pred = tree.predict(X_val)
    tree_results.append((f1_score(y_val, val_pred, zero_division=0), depth, tree))
best_tree_f1, best_tree_depth, best_tree = sorted(tree_results, key=lambda item: (-item[0], item[1]))[0]
model_rows.append({
    "model": f"decision_tree_depth_{best_tree_depth}",
    **score_labels(y_val, best_tree.predict(X_val)),
})

# Default forest on all 17 features.
forest_default = RandomForestClassifier(n_estimators=100, random_state=42)
forest_default.fit(X_train, y_train)
model_rows.append({"model": "random_forest_default", **score_labels(y_val, forest_default.predict(X_val))})

(pd.DataFrame(model_rows, columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"])
 .to_csv(OUTPUTS / "part_c_models.csv", index=False, float_format="%.4f"))

# Feature scores: univariate F scores and random-forest feature importances.
all_selector = SelectKBest(score_func=f_classif, k="all").fit(X_train, y_train)
f_scores = np.nan_to_num(all_selector.scores_, nan=0.0, posinf=0.0, neginf=0.0)
rf_importances = forest_default.feature_importances_
score_table = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances,
    "is_noise": [feature in NOISE_COLUMNS for feature in FEATURES],
})
score_table["f_rank"] = score_table["f_score"].rank(method="min", ascending=False).astype(int)
score_table = score_table[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
score_table.to_csv(OUTPUTS / "part_c_feature_scores.csv", index=False, float_format="%.4f")

# Compare feature-selector sizes; each pipeline is fitted on training rows only.
selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipeline.predict(X_val), zero_division=0)})
pd.DataFrame(selectk_rows).to_csv(OUTPUTS / "part_c_selectk.csv", index=False, float_format="%.4f")

# From this point onward, use k=7. Fit selector on training data only.
selector = SelectKBest(f_classif, k=7)
X_train_7 = selector.fit_transform(X_train, y_train)
X_val_7 = selector.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5
)
grid_search.fit(X_train_7, y_train)

distributions = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=distributions,
    n_iter=9,
    cv=5,
    random_state=42,
)
random_search.fit(X_train_7, y_train)

# Untuned default forest on the same seven selected features.
default_forest_7 = RandomForestClassifier(n_estimators=100, random_state=42)
default_forest_7.fit(X_train_7, y_train)
default_val_f1 = f1_score(y_val, default_forest_7.predict(X_val_7), zero_division=0)

search_rows = []
for search_name, search, candidates, total_fits in [
    ("grid_search", grid_search, 27, 135),
    ("randomized_search", random_search, 9, 45),
]:
    val_f1 = f1_score(y_val, search.best_estimator_.predict(X_val_7), zero_division=0)
    search_rows.append({
        "search": search_name,
        "candidates": candidates,
        "cv_folds": 5,
        "total_fits": total_fits,
        "best_params": json.dumps(search.best_params_, sort_keys=True),
        "best_cv_accuracy": search.best_score_,
        "val_f1": val_f1,
    })
search_rows.append({
    "search": "default_forest_same_7_features",
    "candidates": 1,
    "cv_folds": 0,
    "total_fits": 1,
    "best_params": "defaults",
    "best_cv_accuracy": np.nan,
    "val_f1": default_val_f1,
})
pd.DataFrame(search_rows).to_csv(OUTPUTS / "part_c_tuning.csv", index=False, float_format="%.4f")

# Choose highest validation F1; ties: default, then grid, then randomized.
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
winning_row = sorted(search_rows, key=lambda row: (-row["val_f1"], priority[row["search"]]))[0]
chosen_from = winning_row["search"]
if chosen_from == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_from == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
])
final_model.fit(X_train, y_train)

# Validation threshold table. The positive class is is_delayed == 1.
validation_probabilities = final_model.predict_proba(X_val)[:, 1]
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_rows = []
for threshold in threshold_values:
    val_labels = (validation_probabilities >= threshold).astype(int)
    threshold_rows.append({"threshold": threshold, **score_labels(y_val, val_labels)})
threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUTS / "part_c_thresholds.csv", index=False, float_format="%.4f")

eligible_thresholds = threshold_df.loc[threshold_df["recall"] >= 0.80, "threshold"]
chosen_threshold = float(eligible_thresholds.max()) if len(eligible_thresholds) else 0.1

# Evaluate the test split exactly once at the chosen threshold.
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_labels = (test_probabilities >= chosen_threshold).astype(int)
final_test_metrics = score_labels(y_test, test_labels)
final_test = pd.DataFrame([{"threshold": chosen_threshold, **final_test_metrics}])
final_test.to_csv(OUTPUTS / "part_c_final_test.csv", index=False, float_format="%.4f")

max_depth_value = chosen_params["max_depth"]
config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": "unlimited" if max_depth_value is None else max_depth_value,
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])
config.to_csv(OUTPUTS / "part_c_final_config.csv", index=False)

print("Part C complete")
print(f"Split sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
print(f"Delayed counts: {int(y_train.sum())} / {int(y_val.sum())} / {int(y_test.sum())}")
print(f"Selected tree depth: {best_tree_depth}; selected tuning row: {chosen_from}")
print(f"Chosen alert threshold: {chosen_threshold}")
print(final_test.to_string(index=False))
