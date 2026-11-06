"""Score the 48 new parcels and produce an operations action list."""
from pathlib import Path
import pandas as pd
import joblib

from src.shipment_prep import clean_shipments, build_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

new_raw = pd.read_csv(DATA / "new_shipments.csv")
hubs = pd.read_csv(DATA / "hubs.csv")
model = joblib.load(MODELS / "delay_model.pkl")
threshold = float(pd.read_csv(MODELS / "delay_threshold.csv").loc[0, "threshold"])
feature_order = pd.read_csv(MODELS / "feature_order.csv")["feature"].tolist()
median_table = pd.read_csv(MODELS / "impute_values.csv")
impute_values = dict(zip(median_table["column"], median_table["median"].astype(float)))
hub_segments = pd.read_csv(OUTPUTS / "hub_segments.csv")[["hub_id", "segment"]]

clean_new, _ = clean_shipments(new_raw, medians=impute_values)
featured = build_features(clean_new, hubs)
X_new = featured[feature_order]
probabilities = model.predict_proba(X_new)[:, 1]

scored = featured[["shipment_id", "hub_id"]].copy()
scored["delay_probability"] = probabilities.round(4)
scored["flagged"] = (probabilities >= threshold).astype(int)
scored = scored.merge(hub_segments, on="hub_id", how="left", validate="many_to_one")
scored["action"] = "No action"
flagged = scored["flagged"].eq(1)
escalate = flagged & scored["segment"].eq("Escalate")
proactive = flagged & ~scored["segment"].eq("Escalate")
scored.loc[proactive, "action"] = "Proactive customer call"
scored.loc[escalate, "action"] = "Escalate to hub manager"

output = scored[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
output.to_csv(OUTPUTS / "new_shipment_actions.csv", index=False)
print("Scoring complete")
print(f"New parcels: {len(output)}")
print(f"Flagged: {int(output['flagged'].sum())}")
print(f"Hub-manager escalations: {(output['action'] == 'Escalate to hub manager').sum()}")
