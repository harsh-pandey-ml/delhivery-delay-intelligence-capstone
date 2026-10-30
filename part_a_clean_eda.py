
import os 
import numpy as np 
import pandas as pd 
from src.shipment_prep import clean_shipments, build_features 

os.makedirs("outputs", exist_ok=True) 

raw = pd.read_csv("/content/data/raw_shipments.csv")
hubs = pd.read_csv("/content/data/hubs.csv")

#calculate initial data quality 
raw_rows = len(raw)
duplicate_rows_removed = int(raw.duplicated().sum())
rows_after_dedup = len(raw.drop_duplicates())
missing_distance_before = int(raw["distance_km"].isna().sum())

missing_declared_value_before = int(raw["declared_value"].isna().sum())

missing_payment_mode_before = int(raw["payment_mode"].isna().sum()) 

missing_parcel_weight_before = int(raw["parcel_weight_kg"].isna().sum())

service_type_spelling_before = int(raw.drop_duplicates()["service_type"].nunique())

cleaned, medians = clean_shipments(raw)
features = build_features(cleaned, hubs)

features["is_delayed"] = (features["delivery_hours"] > features["promised_hours"]).astype(int) 

features.to_csv("data/modelling_table.csv", index = False) 

sentinal_weight_rows = int(raw.drop_duplicates()["parcel_weight_kg"].gt(100).sum())
delayed_count = int(features["is_delayed"].sum())
delay_rate = features["is_delayed"].mean()

# delivery hours outliers 
q1 = np.percentile(features["delivery_hours"],25)
q3 = np.percentile(features["delivery_hours"],75) 

#Inter Quartile Range
iqr = q3 - q1 
 
#lower range and upper range 
lower_bound =  q1 - 1.5*iqr 
upper_bound = q3 + 1.5*iqr 

hours_outlier_count = int(features[(features["delivery_hours"] < lower_bound) | (features["delivery_hours"] > upper_bound)]["delivery_hours"].count())

declared_value_mean_inr = features["declared_value_inr"].mean()
declared_value_median_inr = features["declared_value_inr"].median()
log_declared_value_mean = features["log_declared_value"].mean()
log_declared_value_median = features["log_declared_value"].median() 

checks = { 
    "raw_rows" : len(raw),
    "duplicated_rows_removed" : int(raw.duplicated().sum()),
    "rows_after_dedup" : len(cleaned),
    "sentinal_weight_rows" : sentinal_weight_rows,
    "missing_distance_before" : missing_distance_before,
    "missing_declared_value_before" : missing_declared_value_before,
    "missing_payment_mode_before" : missing_payment_mode_before,
    "delayed_count" : delayed_count, 
    "delay_rate" : delay_rate,
    "q1_delivery_hours" : q1,
    "q3_delivery_hours" : q3,
    "iqr_delivery_hours" : iqr,
    "lower_bound_delivery_hours" : lower_bound,
    "upper_bound_delivery_hours" : upper_bound,
    "hours_outlier_count" : hours_outlier_count,
    "declared_value_mean_inr" : declared_value_mean_inr,
    "declared_value_median_inr" : declared_value_median_inr,
    "log_declared_value_mean" : log_declared_value_mean,
    "log_declared_value_median" : log_declared_value_median,
    "service_type_spelling_before" : service_type_spelling_before,
}

checks_df = pd.DataFrame(
    list(checks.items()), 
    columns= ["metric", "value"]
)

checks_df.to_csv("outputs/part_a_checks.csv", index=False)


# delay by group report 

service_summary = (
    features.groupby("service_type")
    .agg(
    shipments = ("is_delayed", "size"),
    delayed = ("is_delayed", "sum"),
    delay_rate = ("is_delayed","mean"),
    mean_hours = ("delivery_hours", "mean"),
    ).reset_index()
)

tier_summary = (
    features.groupby("hub_tier")
    .agg(
    shipments = ("is_delayed", "size"),
    delayed = ("is_delayed", "sum"),
    delay_rate = ("is_delayed","mean"),
    mean_hours = ("delivery_hours", "mean"),
    ).reset_index()
)

features["distance_band"] = pd.cut(
    features["distance_km"],
    bins = [0, 200, 400, 600, 2000],
    labels = ["0-200", "201-400", "401-600", "601+"],
    right = True,
)

distance_summary = (
    features.groupby("distance_band", observed = False)
    .agg(
    shipments = ("is_delayed", "size"),
    delayed = ("is_delayed", "sum"),
    delay_rate = ("is_delayed","mean"),
    mean_hours = ("delivery_hours", "mean"),
    ).reset_index()
)

# service,tier, distance have diff group column name so we rename them
service_summary = service_summary.rename(
    columns = {"service_type": "group"}
)
tier_summary = tier_summary.rename(
    columns = {"hub_tier": "group"}
)
distance_summary = distance_summary.rename(
    columns = {"distance_band": "group"}
)

# adding dimension name to each summary 
service_summary["dimension"] = "service_type"
tier_summary["dimension"] = "hub_tier"
distance_summary["dimension"] = "distance_band" 

# combining them vertically 
delay_by_group = pd.concat(
    [service_summary, tier_summary, distance_summary],
    ignore_index = True
)

# now arranging the column in a required order 
delay_by_group = delay_by_group[
    ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
]

delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index = False)


# correlation matrix :- measures the strength and direction of linear relationship b/w 2 variable

numeric_features = features.select_dtypes(
    include = "number"
)
correlations = numeric_features.corr()

delivery_corr = correlations["delivery_hours"].drop(labels = ["delivery_hours"])

top_three = delivery_corr.loc[
    delivery_corr.abs().sort_values(ascending = False).head(3).index
    ]
print(top_three.round(4))

correlations.to_csv("outputs/part_a_correlations.csv", index = True)
# index=true preserves row label
