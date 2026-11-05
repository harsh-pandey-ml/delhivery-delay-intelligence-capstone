
import pandas as pd
import numpy as np
import joblib

from src.shipment_prep import clean_shipments, build_features

# OBJECTIVE:- in this part , we have to do the prediction for the new_shipment

# first, we will load the new_shipment file

new_shipments = pd.read_csv("/content/data/new_shipments.csv")
hubs = pd.read_csv("/content/data/hubs.csv")

model = joblib.load("/content/models/delay_model.pkl")  # load the model

threshold_df = pd.read_csv("/content/models/delay_threshold.csv") #load the threshold
threshold = float(threshold_df.loc[0, "threshold"])

feature_order_df = pd.read_csv("/content/models/feature_order.csv")  # load features
FEATURES = feature_order_df["feature"].tolist()

impute_values_df = pd.read_csv("/content/models/impute_values.csv") # load impute values
medians = dict(
    zip(
        impute_values_df["column"],
        impute_values_df["median"]
    )
)

# now clean the new_shipment file and build features
clean_df, medians = clean_shipments(new_shipments, medians=medians)
features_df = build_features(clean_df, hubs)

X_new = features_df[FEATURES]  # now we will arrange the features build in new_features in saved order so that our model dont raise any error

delay_probabilities = model.predict_proba(X_new)[:,1]  # predicting the delay probabilities for the new_shipment order
features_df["delay_probability"] = np.round(delay_probabilities, 4)

features_df["flagged"] = (features_df["delay_probability"] >= threshold) # now we will flag the deliveries which do not satisfy our threshold criteria

hub_segments = pd.read_csv("/content/outputs/hub_segments.csv")

features_df = features_df.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left", validate = "many_to_one") #here we merged the hub_id and segment to the features_df

def recommended_action(row):
  if row["flagged"] and row["segment"] == "Escalate":
    return "Escalate to hub manager"
  if row["flagged"]:
    return "Proactive customer call"
  else:
    return "No action"

features_df["action"] = features_df.apply(recommended_action, axis=1)

output_df = features_df[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]

output_df.to_csv("/content/outputs/new_shipment_actions.csv", index=False)

print("\nNew shipments prediction completed successfully.")

print("\nNumber of shipments:", len(output_df))
print("Number of flagged shipments:", (output_df["flagged"]).sum())

print("Action recommended:")
print(output_df["action"].value_counts())

print("\nPreview:")
print(output_df.head())

print("\nSaved files:")
print("/content/outputs/new_shipment_actions.csv")
