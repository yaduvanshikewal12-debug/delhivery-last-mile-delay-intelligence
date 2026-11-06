"""Shared data-cleaning and feature-building functions for the capstone."""
from __future__ import annotations

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]


def clean_shipments(raw: pd.DataFrame, medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean a raw shipment export and return (clean dataframe, imputation medians)."""
    clean = raw.copy()

    # 1. Remove exact duplicate rows.
    clean = clean.drop_duplicates().copy()

    # 2. Standardize the service names.
    clean["service_type"] = clean["service_type"].astype("string").str.strip().str.title()

    # 3. Make missing payment mode an explicit category.
    clean["payment_mode"] = clean["payment_mode"].fillna("Unknown")

    # 4. Parse rupee-formatted declared values into numeric INR.
    declared = clean["declared_value"].astype("string")
    declared = declared.str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    clean["declared_value_inr"] = pd.to_numeric(declared, errors="coerce").astype(float)
    clean = clean.drop(columns=["declared_value"])

    # 5. Parse dispatch timestamp.
    clean["dispatch_time"] = pd.to_datetime(clean["dispatch_time"], errors="coerce")

    # 6. Treat scanner-fault sentinel weights as missing.
    clean.loc[clean["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Compute historical medians once, or reuse the stored historical values.
    median_columns = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        used_medians = {column: float(clean[column].median()) for column in median_columns}
    else:
        used_medians = {column: float(medians[column]) for column in median_columns}

    for column in median_columns:
        clean[column] = clean[column].fillna(used_medians[column])

    return clean, used_medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub attributes and create the pinned model features."""
    hub_columns = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    result = clean.merge(hubs[hub_columns], on="hub_id", how="left", validate="many_to_one")

    result["promised_hours"] = result["service_type"].map(PROMISED_HOURS).astype(int)
    result["is_cod"] = (result["payment_mode"] == "COD").astype(int)
    result["is_weekend"] = (result["dispatch_time"].dt.dayofweek >= 5).astype(int)
    result["dispatch_hour"] = result["dispatch_time"].dt.hour.astype(int)
    result["log_declared_value"] = np.log(result["declared_value_inr"].astype(float))
    return result
