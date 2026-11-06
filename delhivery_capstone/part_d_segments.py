"""Part D: profile hubs, cluster them into operating segments and compare tier labels."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

modelling = pd.read_csv(DATA / "modelling_table.csv")

profile = (modelling.groupby("hub_id", as_index=False, sort=True)
           .agg(
               shipments=("shipment_id", "count"),
               mean_delivery_hours=("delivery_hours", "mean"),
               delay_rate=("is_delayed", "mean"),
               mean_distance_km=("distance_km", "mean"),
               mean_n_stops=("n_stops", "mean"),
           )
           .sort_values("hub_id", kind="stable")
           .reset_index(drop=True))
profile.to_csv(OUTPUTS / "hub_profile.csv", index=False, float_format="%.4f")

profile_features = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
scaled = StandardScaler().fit_transform(profile[profile_features])

# K-means separation ratios using all 276 unique unordered hub pairs.
def separation_ratio(matrix, labels):
    within_distances = []
    across_distances = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            distance = float(np.linalg.norm(matrix[i] - matrix[j]))
            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)
    return float(np.mean(across_distances) / np.mean(within_distances))

separation_rows = []
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled)
    sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
    separation_rows.append({
        "K": k,
        "across_over_within": separation_ratio(scaled, labels),
        "cluster_sizes": "-".join(str(size) for size in sizes),
    })
pd.DataFrame(separation_rows).to_csv(OUTPUTS / "part_d_separation.csv", index=False, float_format="%.4f")

# The business requires K=3 playbooks. Name clusters by their observed delay rate.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
profile["cluster"] = kmeans.fit_predict(scaled)
cluster_order = (profile.groupby("cluster")["delay_rate"].mean()
                 .sort_values(kind="stable").index.tolist())
segment_names = {cluster_order[0]: "Standard", cluster_order[1]: "Watch", cluster_order[2]: "Escalate"}
profile["segment"] = profile["cluster"].map(segment_names)

hubs = pd.read_csv(DATA / "hubs.csv")
hub_segments = profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one"
)
hub_segments.to_csv(OUTPUTS / "hub_segments.csv", index=False)

segment_profile = (profile.groupby("segment", as_index=False)
                   .agg(
                       hubs=("hub_id", "count"),
                       mean_delivery_hours=("mean_delivery_hours", "mean"),
                       delay_rate=("delay_rate", "mean"),
                       mean_distance_km=("mean_distance_km", "mean"),
                       mean_n_stops=("mean_n_stops", "mean"),
                   ))
segment_order = ["Standard", "Watch", "Escalate"]
segment_profile["segment"] = pd.Categorical(segment_profile["segment"], categories=segment_order, ordered=True)
segment_profile = segment_profile.sort_values("segment").reset_index(drop=True)
segment_profile["segment"] = segment_profile["segment"].astype(str)
segment_profile.to_csv(OUTPUTS / "segment_profile.csv", index=False, float_format="%.4f")

implied_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatches = hub_segments.loc[
    hub_segments.apply(lambda row: row["segment"] != implied_segment[int(row["hub_tier"])], axis=1)
].copy()
mismatches.to_csv(OUTPUTS / "tier_vs_segment_mismatch.csv", index=False)

print("Part D segmentation complete")
print(segment_profile.to_string(index=False))
print("Tier mismatches:")
print(mismatches.to_string(index=False))
