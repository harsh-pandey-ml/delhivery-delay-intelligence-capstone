
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# first of all, we will read the data from modelling and hub table and create the hub profile and save the output
df = pd.read_csv("/content/data/modelling_table.csv")
hubs = pd.read_csv("/content/data/hubs.csv")

print("Dataset shape:", df.shape)
print("Hubs shape:", hubs.shape)

print("No.of hubs", hubs["hub_id"].nunique())

hub_profile = df.groupby("hub_id").agg( 
    shipments = ("shipment_id", "count"),
    mean_delivery_hours = ("delivery_hours", "mean"),
    delay_rate = ("is_delayed", "mean"),
    mean_distance_km = ("distance_km", "mean"),
    mean_n_stops = ("n_stops", "mean") 
).reset_index() 

hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True) 
hub_profile.to_csv("outputs/hub_profiles.csv", index=False) 

print("\nHub profile:")
print(hub_profile.head())


#now we will scale four profile columns 

profile_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
    ]

X = hub_profile[profile_features] 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

#now train a new KMean clustering model for different values and predict the cluster label 

separation_rows = []

for k in [2,3,4,5]:
  kmeans = KMeans(
      n_clusters = k,
      random_state = 42,
      n_init = 10
  ) 

  cluster_labels = kmeans.fit_predict(X_scaled) 
  cluster_sizes = pd.Series(cluster_labels).value_counts()  # now we are predicting that how many clusters are there and how many values they store
  cluster_sizes = sorted(cluster_sizes.tolist(), reverse=True) # we will get the list of clusters 
  cluster_sizes_test = "-".join(str(size) for size in cluster_sizes) # seprate them by "-"
  
  # now calculate the across cluster and within cluster distance 
  within_distances = []
  across_distances = []

  for i in range(len(X_scaled)):
    for j in range(i+1, len(X_scaled)):
      distance = np.linalg.norm(X_scaled[i] - X_scaled[j])
      if cluster_labels[i] == cluster_labels[j]:
        within_distances.append(distance)
      else:
        across_distances.append(distance) 
  
  mean_within = np.mean(within_distances)
  mean_across = np.mean(across_distances)

  across_over_within = mean_across / mean_within 

  separation_rows.append({
    "K" : k,
    "across_over_within" : across_over_within,
    "cluster_sizes" : cluster_sizes_test
  })

separation_df = pd.DataFrame(separation_rows)
separation_df.to_csv("outputs/part_d_separation.csv", index=False)

print("\nSeparation result:")
print(separation_df)


#fitting final kmeans model with k=3

kmeans = KMeans(
    n_clusters = 3,
    random_state = 42,
    n_init = 10
) 
hub_profile["cluster"] = kmeans.fit_predict(X_scaled)


# now we will name the segment based on delay rate 

cluster_delay_rate = hub_profile.groupby("cluster")["delay_rate"].mean()
cluster_delay_rate = cluster_delay_rate.sort_values()

segment_names = {
    cluster_delay_rate.index[0] : "Standard",
    cluster_delay_rate.index[1] : "Watch",
    cluster_delay_rate.index[2] : "Escalate"
}
hub_profile["segment"] = hub_profile["cluster"].map(segment_names) 

# creating hub segment profile
hub_segments = hub_profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]
hub_segments = hub_segments.sort_values("hub_id") 
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

print("\nHub segments:")
print(hub_segments)


# now create a default segment profile from hub_profile 

segment_profile = hub_profile.groupby("segment").agg(
    hubs = ("hub_id", "count"),
    mean_delivery_hours = ("mean_delivery_hours", "mean"),
    delay_rate = ("delay_rate", "mean"),
    mean_distance_km = ("mean_distance_km", "mean"),
    mean_n_stops = ("mean_n_stops", "mean")
).reset_index()

segment_order = ["Standard", "Watch", "Escalate"]
segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories = segment_order,
    ordered = True
)
segment_profile = segment_profile.sort_values("segment").reset_index(drop=True)  
segment_profile.to_csv("outputs/segment_profile.csv", index=False)

print("\nSegment profiles:")
print(segment_profile)

tier_segment_mapping = {
    1 : "Standard",
    2 : "Watch",
    3 : "Escalate"
}

hub_segments["expected_segment"] = hub_segments["hub_tier"].map(tier_segment_mapping)
mismatch_df = hub_segments[hub_segments["segment"] != hub_segments["expected_segment"]]
mismatch_df = mismatch_df[["hub_id", "segment", "hub_city", "hub_tier"]]
mismatch_df.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print("\nTier vs segment mismatch:")
print(mismatch_df)
