
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

def clean_shipments(raw, medians=None):
  clean_df = raw.copy()

  clean_df = clean_df.drop_duplicates().copy()

  clean_df["service_type"] = (
      clean_df["service_type"]
      .astype("string")
      .str.strip()
      .str.title()
      )

  clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

  clean_df["declared_value_inr"] = pd.to_numeric(
      clean_df["declared_value"]
      .astype("string")
      .str.replace("₹","", regex=False)
      .str.replace(",","", regex=False),
      errors="coerce",
      )

  clean_df = clean_df.drop(columns=["declared_value"])

  clean_df["dispatch_time"] = pd.to_datetime(
      clean_df["dispatch_time"],
      format = "%Y-%m-%d %H:%M",
      errors="coerce",
      )

  clean_df.loc[
      clean_df["parcel_weight_kg"] > 100,
      "parcel_weight_kg"
  ] = np.nan

  median_columns = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
  
  for column in median_columns:
    clean_df[column] = pd.to_numeric(
        clean_df[column],
        errors = "coerce"
        ).astype("float64")

  if medians is None:
    medians = {
        column: round(float(clean_df[column].median()), 2)
        for column in median_columns
    }

  for column in median_columns:
    clean_df[column] = clean_df[column].fillna(medians[column])

  return clean_df, medians


def build_features(clean, hubs):
  features = clean.copy()

  features = features.merge(
      hubs[
          ["hub_id","hub_city", "region", "hub_tier", "hub_daily_capacity"]
      ],
      how="left",
      on="hub_id",
      validate = "many_to_one"
  )

  features["promised_hours"] = (
      features["service_type"]
      .map(PROMISED_HOURS)
      .astype(int)
      )

  features["is_cod"] = (features["payment_mode"] == "COD").astype(int)

  features["is_weekend"] = (features["dispatch_time"].dt.dayofweek >= 5).astype(int)

  features["dispatch_hour"] = (features["dispatch_time"].dt.hour.astype(int))

  features["log_declared_value"] = np.log(features["declared_value_inr"])

  return features
