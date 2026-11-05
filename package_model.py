
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import clean_shipments, build_features

raw = pd.read_csv("/content/data/raw_shipments.csv")
hubs = pd.read_csv("/content/data/hubs.csv")

config = pd.read_csv("/content/outputs/part_c_final_config.csv")

selector_k = int(config.loc[0, "selector_k"])
n_estimators = int(config.loc[0, "n_estimators"])
min_samples_leaf = int(config.loc[0, "min_samples_leaf"])
threshold = float(config.loc[0,"threshold"])

max_depth_value = config.loc[0, "max_depth"]
if max_depth_value == "unlimited":
  max_depth = None
else:
  max_depth = int(max_depth_value)

print("\nFinal configuaration")
print(config)


# now clean the historical shipment data

clean_df, medians = clean_shipments(raw)
features_df = build_features(clean_df, hubs)

features_df["is_delayed"] = (features_df["delivery_hours"] > features_df["promised_hours"]).astype(int)

print("\nCleaned data shapes:", clean_df.shape)
print("Features data shape:", features_df.shape)
print("Median values:", medians)

# 17 features
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

X = features_df[FEATURES]
y = features_df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("\nTrain shapes:", X_train.shape, y_train.shape)
print("Validation shapes:", X_val.shape, y_val.shape)
print("Test shapes:", X_test.shape, y_test.shape)

model = Pipeline([
    ("selector", SelectKBest(f_classif, k=selector_k)),
    ("classifier", RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_samples_leaf = min_samples_leaf,
        random_state = 42
    ))
])

model.fit(X_train, y_train)

# creating the models folder
os.makedirs("/content/models", exist_ok=True)
os.makedirs("/content/outputs", exist_ok=True)

joblib.dump(model, "/content/models/delay_model.pkl")   # saved the model

threshold_df = pd.DataFrame({
    "threshold" : [threshold]
})
threshold_df.to_csv("/content/models/delay_threshold.csv", index=False)

feature_order_df = pd.DataFrame({
    "feature" : FEATURES
})
feature_order_df.to_csv("/content/models/feature_order.csv", index=False)

impute_values_df = pd.DataFrame({
    "column" : list(medians.keys()),
    "median" : list(medians.values())
})
impute_values_df.to_csv("/content/models/impute_values.csv", index=False)

# reloading the save model
loaded_model = joblib.load("/content/models/delay_model.pkl")

# comparing loaded and original probabilities
original_probabilities = model.predict_proba(X_test)[:,1]
reloaded_probabilities = loaded_model.predict_proba(X_test)[:,1]

# calculate their difference
max_diff = np.max(np.abs(original_probabilities - reloaded_probabilities))

# saving the reloaded model check results
reloaded_check_df = pd.DataFrame({
    "test_rows_scored" : [len(X_test)],
    "reloaded_max_abs_probability_diff" : [max_diff],
    "threshold" : [threshold]
})
reloaded_check_df.to_csv("/content/outputs/part_d_reloaded_check.csv", index=False)

print("Model package completed successfully.")

print("\nDataset split:")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

print("\nModel configuration:")
print("Selector k:", selector_k)
print("N estimators:", n_estimators)
print("Max depth:", max_depth_value)
print("Min samples leaf:", min_samples_leaf)
print("Threshold:", threshold)

print("\nReloaded model check:")
print("Test rows scored:", len(X_test))
print("Max abs probability diff:", max_diff)

print("\nSaved files:")
print("Model:", "/content/models/delay_model.pkl")
print("Threshold:", "/content/models/delay_threshold.csv")
print("Feature order:", "/content/models/feature_order.csv")
print("Impute values:", "/content/models/impute_values.csv")
print("Reloaded check:", "/content/outputs/part_d_reloaded_check.csv")
