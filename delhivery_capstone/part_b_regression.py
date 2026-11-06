"""Part B: fit and compare regression models for delivery-hours prediction."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.shipment_prep import FEATURES

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

modelling = pd.read_csv(DATA / "modelling_table.csv")
X = modelling[FEATURES]
y = modelling["delivery_hours"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standalone leakage check: fit only on training rows.
scaler_check = StandardScaler()
X_train_scaled = scaler_check.fit_transform(X_train)
X_test_scaled = scaler_check.transform(X_test)
checks = pd.DataFrame([
    ("train_rows", len(X_train)),
    ("val_rows", len(X_val)),
    ("test_rows", len(X_test)),
    ("train_scaled_max_abs_mean", float(np.abs(X_train_scaled.mean(axis=0)).max())),
    ("test_scaled_max_abs_mean", float(np.abs(X_test_scaled.mean(axis=0)).max())),
], columns=["metric", "value"])
checks.to_csv(OUTPUTS / "part_b_checks.csv", index=False, float_format="%.4f")


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


# Mean baseline.
baseline_pred = np.full(len(y_val), float(y_train.mean()))
metric_rows = [{"model": "mean_baseline", "split": "val", **regression_metrics(y_val, baseline_pred)}]

# Tree-depth diagnostic.
depths = [1, 2, 3, 4, 6, 8, 12, None]
depth_rows = []
tree_models = {}
for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)
    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    val_rmse = mean_squared_error(y_val, val_pred) ** 0.5
    depth_rows.append({"max_depth": "unlimited" if depth is None else depth,
                       "train_rmse": train_rmse, "val_rmse": val_rmse})
    tree_models[depth] = tree
pd.DataFrame(depth_rows).to_csv(OUTPUTS / "part_b_tree_depth.csv", index=False, float_format="%.4f")

best_tree_depth = sorted(depths, key=lambda d: (mean_squared_error(y_val, tree_models[d].predict(X_val)) ** 0.5,
                                                 float("inf") if d is None else d))[0]
best_tree = tree_models[best_tree_depth]

# Four candidates on the same training/validation split.
models = {
    "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "decision_tree": best_tree,
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}
for name, model in models.items():
    if name != "decision_tree":
        model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    metric_rows.append({"model": name, "split": "val", **regression_metrics(y_val, prediction)})

# Learning-rate experiment.
lr_rows = []
for learning_rate in [0.01, 0.1, 1.0]:
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    lr_rows.append({"learning_rate": learning_rate, "val_rmse": mean_squared_error(y_val, pred) ** 0.5,
                    "val_r2": r2_score(y_val, pred)})
pd.DataFrame(lr_rows).to_csv(OUTPUTS / "part_b_gb_learning_rate.csv", index=False, float_format="%.4f")

# Select the two validation-RMSE leaders and compare them with the same KFold object.
ranked_names = sorted(models, key=lambda name: next(r["RMSE"] for r in metric_rows if r["model"] == name))
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name in ranked_names[:2]:
    cv_model = models[name]
    scores = cross_val_score(cv_model, X_train_val, y_train_val, cv=kfold)
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})
pd.DataFrame(cv_rows).to_csv(OUTPUTS / "part_b_cv.csv", index=False, float_format="%.4f")

# Final candidate is chosen by validation RMSE; test is evaluated once.
winning_name = ranked_names[0]
winning_model = models[winning_name]
test_pred = winning_model.predict(X_test)
metric_rows.append({"model": winning_name, "split": "test", **regression_metrics(y_test, test_pred)})
metrics_df = pd.DataFrame(metric_rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"])
metrics_df.to_csv(OUTPUTS / "part_b_metrics.csv", index=False, float_format="%.4f")

print("Part B complete")
print(f"Split sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
print(f"Best tree depth: {best_tree_depth}")
print(f"Validation winner: {winning_name}")
print(metrics_df.to_string(index=False))
