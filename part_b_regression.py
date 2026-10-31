
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("/content/data/modelling_table.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
TARGET = "delivery_hours"

df = pd.read_csv(DATA_PATH) 
required_columns = FEATURES + [TARGET] 
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns: 
  raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any(): 
  raise ValueError(
      "Missing values found in FEATURES or TARGET."
      "check part A before continuing."
  )

X = df[FEATURES].copy()
y = df[TARGET].copy()

print("Dataset shape:", df.shape)
print("Number of features:", len(FEATURES))
print("Feature order:", FEATURES) 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nSplit size:")                                              
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

assert len(X_train) == 840   # validating our values if they are correct then proceed. assert is like a checkpoint 
assert len(X_val) == 180
assert len(X_test) == 180 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_scaled_max_abs_mean = float(np.abs(X_train_scaled.mean(axis=0)).max()) 
test_scaled_max_abs_mean = float(np.abs(X_test_scaled.mean(axis=0)).max())  

checks = pd.DataFrame(
    [
       ("train_rows", len(X_train)),
       ("val_rows", len(X_val)),
       ("test_rows", len(X_test)),
       ("train_scaled_max_abs_mean", train_scaled_max_abs_mean),
       ("test_scaled_max_abs_mean", test_scaled_max_abs_mean),
    ], 
    columns=["metric", "value"],
)  

checks.to_csv(OUTPUT_DIR / "part_b_checks.csv", index=False) 

# NOW WE DO SCALING CHECK 
print("\nScaling checks:")
print(checks.to_string(index=False)) 

def calculate_metrics(y_true, y_pred):
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)
  
  return {
      "MAE" : mae,
      "MSE" : mse,
      "RMSE" : rmse,
      "R2" : r2,
  } 

metrics_rows = []
baseline_prediction = np.full(shape=len(y_val), fill_value = y_train.mean())
baseline_metrics = calculate_metrics(y_val, baseline_prediction)
metrics_rows.append({
    "model": "mean_baseline",
    "split" : "val",
    **baseline_metrics,
})
print("\nBaseline prediction metrics:")
print(baseline_metrics) 

# tree depth diagnosis 
depth = [1,2,3,4,6,8,12, None] 
tree_depth_rows = []
for max_depth in depth:
  tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
  tree.fit(X_train, y_train)
  train_prediction = tree.predict(X_train)
  val_prediction = tree.predict(X_val)
  train_rmse = np.sqrt(mean_squared_error(y_train, train_prediction))
  val_rmse = np.sqrt(mean_squared_error(y_val, val_prediction))
  tree_depth_rows.append({
      "max_depth" : "unlimited" if max_depth is None else max_depth,
      "train_rmse" : float(train_rmse),
      "val_rmse" : float(val_rmse),
  })

tree_depth_df = pd.DataFrame(tree_depth_rows)
tree_depth_df.to_csv(OUTPUT_DIR / "part_b_tree_depth.csv", index=False)

print("\n tree depth experiment:")
print(tree_depth_df.to_string(index=False))

# now we have to chooose lowest rmse which is connected to the smaller depth 
tree_depth_df["tie_break"] = tree_depth_df["max_depth"].map( 
    lambda value: float("inf") if value == "unlimited" else int(value)
)
best_tree_row = tree_depth_df.sort_values(
    ["val_rmse", "tie_break"],
    ascending = [True, True],
    ).iloc[0]

best_depth_value = best_tree_row["max_depth"]

best_depth = (None if best_depth_value=="unlimited" else int(best_depth_value))

print("\nSelected decision tree depth:", best_depth_value)

# now we have to create four candidate model 
models = {
    "linear_regression": Pipeline(
        [

           ("scaler" , StandardScaler()),
           ("model" , LinearRegression())                        
        ]
    ), 
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

fitted_models = {}
validation_rmse = {}

for name, model in models.items():
  model.fit(X_train, y_train)
  fitted_models[name] = model 
  
  prediction = model.predict(X_val)
  model_metrics = calculate_metrics(y_val, prediction)
  validation_rmse[name] = model_metrics["RMSE"]
  metrics_rows.append({
      "model" : name,
      "split" : "val",
      **model_metrics,
  })

print(f"\n{name} validation metrics:")
print(model_metrics) 

# now we have to do the learning rate experiment according to statement 6 of part B of capstone project 
learning_rate_rows = []

for lr in [0.01, 0.1, 1.0]: 
  gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
  gb_model.fit(X_train, y_train)
  prediction = gb_model.predict(X_val)
  gb_metrics = calculate_metrics(y_val, prediction)
  learning_rate_rows.append({
      "learning_rate" : lr,
      "val_rmse" : gb_metrics["RMSE"],
      "val_r2" : gb_metrics["R2"],
  })

learning_rate_df = pd.DataFrame(learning_rate_rows)
learning_rate_df.to_csv(OUTPUT_DIR / "part_b_gb_learning_rate.csv", index=False)

print("\n Gradient boosting learning rate experiment:")
print(learning_rate_df.to_string(index=False))

# now we have to rank the four candidate model 
ranking = sorted(validation_rmse.items(), key=lambda item: (item[1], item[0])) 

print("\nValidation ranking - lowest RMSE is best:")

for position, (name, rmse) in enumerate(ranking, start=1):
  print(f"{position}. {name}: RMSE = {rmse:.4f}hours")

# after ranking , we have to apply the 2 best candidate model on the training and validation rows
# first merge training and validation rows 

x_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0) 

# now apply the same kfold on both 
kf = KFold(n_splits=5, shuffle=True, random_state=42) 
cv_rows = []

for name, _ in ranking[:2]: 
  cv_model = clone(models[name])
  scores = cross_val_score(cv_model, 
                           x_trainval,
                           y_trainval, 
                           cv=kf,
                           ) 
  cv_rows.append({
      "model" : name,
      "mean_r2" : float(scores.mean()),
      "std_r2" : float(scores.std()),
  })

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUTPUT_DIR / "part_b_cv.csv", index=False)

print(f"\nFive fold cross validation:") 
print(cv_df.to_string(index=False))

# now we have to select the final model with lowest validation RMSE and have to apply it on test data once 

final_model_name = ranking[0][0]
final_model = fitted_models[final_model_name] 

test_prediction = final_model.predict(X_test)
test_metrics = calculate_metrics(y_test, test_prediction)

metrics_rows.append({
    "model" : final_model_name,
    "split" : "test",
    **test_metrics,
}) 
metrics_df = pd.DataFrame(
    metrics_rows,
    columns = ["model", "split", "MAE", "MSE", "RMSE", "R2"]
) 
metrics_df.to_csv(OUTPUT_DIR / "part_b_metrics.csv", index=False)

print("\nAll model metrics:")
print(metrics_df.to_string(index=False)) 


# now perform the acceptance checks 

assert len(tree_depth_df) == 8
assert len(learning_rate_df) == 3
assert len(cv_df) == 2 
assert len(metrics_df) ==6 

assert train_scaled_max_abs_mean < 1e-9   #scaling checks
assert test_scaled_max_abs_mean >0.01  

assert np.allclose(                  #RMSE must be equal to sqrt(mse)
    metrics_df["RMSE"],
    np.sqrt(metrics_df["MSE"]),
    atol = 0.001,
)

decision_tre_val_rmse = metrics_df.loc[
    (metrics_df["model"] == "decision_tree") & (metrics_df["split"] == "val"),
    "RMSE"
].iloc[0] 

assert abs(decision_tre_val_rmse - best_tree_row["val_rmse"]) <=0.01 

finite_depths = tree_depth_df[
    tree_depth_df["max_depth"] != "unlimited"
].copy()

finite_depths["max_depth"] = finite_depths["max_depth"].astype(int)
finite_depths = finite_depths.sort_values("max_depth") 

train_errors = finite_depths["train_rmse"].to_numpy() 

assert np.all(np.diff(train_errors) <=1e-9) 

lr_r2 = learning_rate_df.set_index("learning_rate")["val_r2"]

assert lr_r2.loc[0.1] > lr_r2.loc[1.0] > lr_r2.loc[0.01] 

assert(cv_df["mean_r2"] >= 0.90).all()        # cross validation performance
assert(cv_df["std_r2"] <= 0.03).all()

assert test_metrics["R2"] >= 0.88             # final test performance
print("created")
print("- outputs/part_b_checks.csv")
print("- outputs/part_b_tree_depth.csv")
print("- outputs/part_b_metrics.csv")
print("- outputs/part_b_gb_learning_rate.csv")
print("- outputs/part_b_cv.csv")

print("\nReview the results and accpetance checks above.")
print("\nAll acceptance checks passed.")
