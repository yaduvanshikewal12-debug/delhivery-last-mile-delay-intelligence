"""Part A: clean the historical data and write EDA evidence tables."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.shipment_prep import FEATURES, PROMISED_HOURS, clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

raw = pd.read_csv(DATA / "raw_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")

# Counts that must be measured after deduplication but before imputation.
dedup = raw.drop_duplicates().copy()
raw_rows = len(raw)
duplicate_rows_removed = raw_rows - len(dedup)
service_spellings_before = int(dedup["service_type"].nunique(dropna=True))
missing_distance_before = int(dedup["distance_km"].isna().sum())
missing_weight_before = int(dedup["parcel_weight_kg"].isna().sum())
missing_value_before = int(dedup["declared_value"].isna().sum())
missing_payment_before = int(dedup["payment_mode"].isna().sum())
sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())

clean, medians = clean_shipments(raw)
service_spellings_after = int(clean["service_type"].nunique(dropna=True))
modelling = build_features(clean, hubs)
modelling["is_delayed"] = (modelling["delivery_hours"] > modelling["promised_hours"]).astype(int)

# Preserve the full cleaned table, including descriptive and target columns.
modelling.to_csv(DATA / "modelling_table.csv", index=False)

# Summary/check metrics.
q1 = float(modelling["delivery_hours"].quantile(0.25))
q3 = float(modelling["delivery_hours"].quantile(0.75))
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outlier_count = int(((modelling["delivery_hours"] < lower_bound) | (modelling["delivery_hours"] > upper_bound)).sum())
log_values = np.log(modelling["declared_value_inr"])
checks = [
    ("raw_rows", raw_rows),
    ("duplicate_rows_removed", duplicate_rows_removed),
    ("rows_after_dedup", len(dedup)),
    ("missing_distance_km_before", missing_distance_before),
    ("missing_parcel_weight_before", missing_weight_before),
    ("missing_declared_value_before", missing_value_before),
    ("missing_payment_mode_before", missing_payment_before),
    ("sentinel_weight_rows", sentinel_weight_rows),
    ("service_type_spellings_before", service_spellings_before),
    ("service_type_spellings_after", service_spellings_after),
    ("median_distance_km", medians["distance_km"]),
    ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
    ("median_declared_value_inr", medians["declared_value_inr"]),
    ("rows_after_hub_merge", len(modelling)),
    ("nulls_in_features_and_target", int(modelling[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())),
    ("delayed_count", int(modelling["is_delayed"].sum())),
    ("delay_rate", float(modelling["is_delayed"].mean())),
    ("hours_q1", q1),
    ("hours_q3", q3),
    ("hours_lower_bound", lower_bound),
    ("hours_upper_bound", upper_bound),
    ("hours_outlier_count", outlier_count),
    ("declared_value_mean_inr", float(modelling["declared_value_inr"].mean())),
    ("declared_value_median_inr", float(modelling["declared_value_inr"].median())),
    ("log_declared_value_mean", float(log_values.mean())),
    ("log_declared_value_median", float(log_values.median())),
]
checks_df = pd.DataFrame(checks, columns=["metric", "value"])
checks_df["value"] = checks_df["value"].map(lambda v: round(v, 4) if isinstance(v, (float, np.floating)) else v)
checks_df.to_csv(OUTPUTS / "part_a_checks.csv", index=False)

# Delay evidence by service, tier and distance band.
rows = []
for dimension, column in [("service_type", "service_type"), ("hub_tier", "hub_tier")]:
    grouped = modelling.groupby(column, dropna=False, sort=True)
    for group, frame in grouped:
        rows.append({
            "dimension": dimension,
            "group": group,
            "shipments": len(frame),
            "delayed": int(frame["is_delayed"].sum()),
            "delay_rate": float(frame["is_delayed"].mean()),
            "mean_hours": float(frame["delivery_hours"].mean()),
        })

band_labels = ["0-200", "201-400", "401-600", "601+"]
modelling["distance_band"] = pd.cut(
    modelling["distance_km"], bins=[0, 200, 400, 600, 2000], labels=band_labels, right=True, include_lowest=True
)
for group, frame in modelling.groupby("distance_band", observed=False, sort=True):
    rows.append({
        "dimension": "distance_band",
        "group": str(group),
        "shipments": len(frame),
        "delayed": int(frame["is_delayed"].sum()),
        "delay_rate": float(frame["is_delayed"].mean()) if len(frame) else 0.0,
        "mean_hours": float(frame["delivery_hours"].mean()) if len(frame) else np.nan,
    })
delay_by_group = pd.DataFrame(rows, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"])
for column in ["delay_rate", "mean_hours"]:
    delay_by_group[column] = delay_by_group[column].round(4)
delay_by_group.to_csv(OUTPUTS / "part_a_delay_by_group.csv", index=False)

# Numeric correlation matrix, with the row labels retained as the first column.
numeric = modelling.select_dtypes(include=[np.number])
numeric.corr().to_csv(OUTPUTS / "part_a_correlations.csv", index=True, index_label="column", float_format="%.4f")

print("Part A complete")
print(f"Rows in modelling table: {len(modelling)}")
print(f"Delayed parcels: {int(modelling['is_delayed'].sum())} ({modelling['is_delayed'].mean():.4f})")
print(f"Imputation medians: {medians}")
