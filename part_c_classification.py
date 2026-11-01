
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = Path("/content/data/modelling_table.csv")
OUTPUT_DIR = Path("/content/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

TARGET = "is_delayed"

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

df = pd.read_csv(DATA_PATH)
required_columns = FEATURES + [TARGET]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
  raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any():
  raise ValueError("Missing values found in FEATURES or TARGET.")

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

print("Dataset shape:", df.shape)
print("Number of features:", len(FEATURES))
print("Overall delayed parcels:", int(y.sum()))

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("\nsplit summary")
print("Training:", len(X_train), "| Delayed:", int(y_train.sum()))
print("Validation:", len(X_val), "| Delayed:", int(y_val.sum()))
print("Test:", len(X_test), "| Delayed:", int(y_test.sum()))

# now cross varify with the values given in task1 of part C
assert len(X_train) == 840
assert len(X_val) == 180
assert len(X_test) == 180
assert int(y_train.sum()) == 195
assert int(y_val.sum()) == 42
assert int(y_test.sum()) == 41

def calculate_metrics(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
  return{
      "TN" : int(tn),
      "FP" : int(fp),
      "FN" : int(fn),
      "TP" : int(tp),
      "accuracy" : accuracy_score(y_true, y_pred),
      "precision" : precision_score(y_true, y_pred, zero_division=0),
      "recall" : recall_score(y_true, y_pred, zero_division=0),
      "f1" : f1_score(y_true, y_pred, zero_division=0),
  }

model_rows = []

blind_predictions = np.zeros(len(y_val), dtype=int)         # create a blind baseline predicting every parcel on time
blind_metrics = calculate_metrics(y_val, blind_predictions)
blind_metrics["precision"] = 0.0   #precision represent how much we correctly got but blind predictions predict every parcel on time so precision=0

model_rows.append({
    "model" : "blind_all_on_time",
    **blind_metrics
})

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_val)

#add logistic model prediction in model_rows
model_rows.append({
    "model" : "logistic_regression",
    **calculate_metrics(y_val, logistic_predictions)
})

tree_depth_rows = []

for depth in [2,3,4,5,6,8,10]:
  tree_model = DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=42)
  tree_model.fit(X_train, y_train)
  predictions = tree_model.predict(X_val)
  metrics = calculate_metrics(y_val, predictions)
  tree_depth_rows.append({
      "depth" : depth,
      "f1" : metrics["f1"]
  })

tree_depth_df = pd.DataFrame(tree_depth_rows)  # converted the tree_depth_rows into dataframe and stored in tree_depth_df

best_tree_row = tree_depth_df.sort_values(     # arranged the rows in order that first row will have high F1 and low depth
    ["f1", "depth"],
    ascending = [False, True]
).iloc[0]
best_tree_depth = int(best_tree_row["depth"])

print("\nDecision tree experiment:")
print(tree_depth_df.to_string(index=False))
print("Selected tree depth:", best_tree_depth)

best_tree = DecisionTreeClassifier(criterion="gini", max_depth=best_tree_depth, random_state=42)
best_tree.fit(X_train, y_train)
tree_predictions = best_tree.predict(X_val)

model_rows.append({
    "model" : f"decision_tree_depth_{best_tree_depth}",
    **calculate_metrics(y_val, tree_predictions)
})

default_forest = RandomForestClassifier(n_estimators=100, random_state=42)
default_forest.fit(X_train, y_train)
forest_predictions = default_forest.predict(X_val)

model_rows.append({
    "model" : "random_forest_default",
    **calculate_metrics(y_val, forest_predictions)
})

model_df = pd.DataFrame(model_rows)
model_df.to_csv(OUTPUT_DIR / "part_c_models.csv", index=False)

print("\n Classification model evaluation:")
print(model_df.to_string(index=False))


# fitting default forest on 17 columns of training rows and getting important feature

importance_forest = RandomForestClassifier(n_estimators=100, random_state=42)
importance_forest.fit(X_train,y_train)

selector_all = SelectKBest(score_func=f_classif, k="all")
selector_all.fit(X_train, y_train)
f_score = selector_all.scores_

feature_score_df = pd.DataFrame({
    "feature" : FEATURES,
    "f_score" : f_score,
    "rf_importance" : importance_forest.feature_importances_,
    "is_noise" : [col in NOISE_COLUMNS for col in FEATURES]
})

feature_score_df["f_rank"] = (
    feature_score_df["f_score"]
    .rank(ascending=False, method="min")      # if two tied features have same f_score we will give them same rank without choosing one over other so we used method=min
    .astype(int)
    )

feature_score_df  = feature_score_df[[
    "feature", "f_score", "f_rank", "rf_importance", "is_noise"
]].sort_values(["f_rank", "feature"]).reset_index(drop=True)

feature_score_df.to_csv(OUTPUT_DIR / "part_c_feature_scores.csv", index=False)

noise_importance_share = feature_score_df.loc[
    feature_score_df["is_noise"],
    "rf_importance"
].sum()

print("\nFeatures score:")
print(feature_score_df.to_string(index=False))
print("\nCombined noise-column forest importance share:", round(noise_importance_share,4))

selectk_rows = []

for k in [5,7,10,12,17]:
  selection_model = Pipeline([
      ("selector" , SelectKBest(f_classif, k=k)),
      ("model" , RandomForestClassifier(
          n_estimators=100,
          random_state=42
          ))
  ])
  selection_model.fit(X_train, y_train)
  predictions = selection_model.predict(X_val)

  selectk_rows.append({
      "k" : k,
      "val_f1" : f1_score(y_val, predictions, zero_division=0)
  })

selectk_df = pd.DataFrame(selectk_rows)

selectk_df.to_csv(OUTPUT_DIR / "part_c_selectk.csv", index=False)

print("\n feature selection experiments:")
print(selectk_df.to_string(index=False)) 


# now we have to select only top 7 features 
selector_7 = SelectKBest(score_func = f_classif, k=7)  

X_train_7 = selector_7.fit_transform(X_train, y_train)  # fit the selected features on training data to learn the pattern
X_val_7 = selector_7.transform(X_val)              # apply those learned pattern on X_val

selected_features = np.array(FEATURES)[selector_7.get_support()].tolist()

print("\nSelected features:")
print(selected_features) 

parameter_grid = {
    "n_estimators" : [50,100,150],
    "max_depth" : [4,6,8],
    "min_samples_leaf" : [1,5,10]
} 

# according to statement 4 of part c ,we applied the grid search and randomized search on tranformed training rows 
base_forest = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator = base_forest, param_grid = parameter_grid, cv=5)
grid_search.fit(X_train_7, y_train) 

grid_predictions = grid_search.best_estimator_.predict(X_val_7) # predicted the grid search over validation data
grid_val_f1 = f1_score(y_val, grid_predictions, zero_division = 0) 

grid_candidates = len(grid_search.cv_results_["params"]) # this gives us best parameter  

random_search = RandomizedSearchCV(
    estimator = RandomForestClassifier(random_state = 42),
    param_distributions = parameter_grid,
    n_iter = 9,
    cv=5,
    random_state=42
    ) 
random_search.fit(X_train_7, y_train) 

random_predictions = random_search.best_estimator_.predict(X_val_7) 
random_val_f1 = f1_score(y_val, random_predictions, zero_division = 0)

random_candidates = len(random_search.cv_results_["params"])

# apply default forest on 7 selected features 
default_forest_7 = RandomForestClassifier(n_estimators=100, random_state=42)
default_forest_7.fit(X_train_7, y_train)
default_7_predictions = default_forest_7.predict(X_val_7)
default_7_val_f1 = f1_score(y_val, default_7_predictions, zero_division = 0) 

tuning_rows = [
    {
        "search" : "grid_search",
        "candidates" : grid_candidates,
        "cv_folds" : 5,
        "total_fits" : grid_candidates*5,
        "best_params" : str(grid_search.best_params_),
        "best_cv_accuracy" : round(grid_search.best_score_,4),
        "val_f1" : round(grid_val_f1,4),
    },
    {
        "search" : "randomized_search",
        "candidates" : random_candidates,
        "cv_folds" : 5,
        "total_fits" : random_candidates*5,
        "best_params" : str(random_search.best_params_),
        "best_cv_accuracy" : round(random_search.best_score_,4),
        "val_f1" : round(random_val_f1,4),
    },
    {
        "search" : "default_forest_same_7_features",
        "candidates" : 1,
        "cv_folds" : 0,
        "total_fits" : 1,
        "best_params" : "defaults", 
        "best_cv_accuracy" : np.nan, 
        "val_f1" : default_7_val_f1
        }
] 

tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(OUTPUT_DIR / "part_c_tuning.csv", index=False)

print("\nTuning experiments:")
print(tuning_df.to_string(index=False)) 

# according to statement 5 of part C, we will select rows with high val_f1 from part_c_tuning (also tie priority) and apply the final modelon training rows 

tie_priority = { 
    "default_forest_same_7_features" : 0,
    "grid_search" : 1,
    "randomized_search" : 2,
}

tuning_df["tie_priority"] = tuning_df["search"].map(tie_priority)
winner = tuning_df.sort_values(
    ["val_f1", "tie_priority"],
    ascending = [False, True]
).iloc[0] 

chosen_from = winner["search"]

if chosen_from == "grid_search": 
  chosen_params = grid_search.best_params_ 
elif chosen_from == "randomized_search":
  chosen_params = random_search.best_params_
else:
  chosen_params = {
      "n_estimators" : 100,
      "max_depth" : None,
      "min_samples_leaf" : 1,
  }

final_model = Pipeline([
    ("selector" , SelectKBest(f_classif, k=7)),
    ("model" , RandomForestClassifier(
        random_state=42, 
        **chosen_params
        ))
])

final_model.fit(X_train, y_train)


# now predict final model's validation probabilities predict delayed for probabilities > threshold

val_probabilities = final_model.predict_proba(X_val)[:,1]
threshold_rows = []
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in threshold_values:
  predictions = (val_probabilities >= threshold).astype(int)

  threshold_rows.append({
    "threshold" : threshold,
    **calculate_metrics(y_val, predictions)
})

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(OUTPUT_DIR / "part_c_thresholds.csv", index=False) 

print("\nValidation threshold experiment:")
print(threshold_df.to_string(index=False)) 

eligible_thresholds = threshold_df[threshold_df["recall"] >= 0.80]
if eligible_thresholds.empty: 
  chosen_threshold = 0.1
else:
  chosen_threshold = float(eligible_thresholds["threshold"].max())

print("\nChosen alert threshold:", chosen_threshold)


# now we will do the evaluation once on the test data  

test_probabilities = final_model.predict_proba(X_test)[:,1]
test_predictions = (test_probabilities >= chosen_threshold).astype(int)

test_metrics = calculate_metrics(y_test, test_predictions)

final_test_df = pd.DataFrame([{
    "threshold" : chosen_threshold,
    **test_metrics
}])
final_test_df.to_csv(OUTPUT_DIR / "part_c_final_test.csv", index=False)

print("\n final test matrics:")
print(final_test_df.to_string(index=False))


#now we will save the final model 

max_depth_value = chosen_params["max_depth"]

final_config_df = pd.DataFrame([{
    "selector_k" : 7,
    "n_estimators" : chosen_params["n_estimators"],
    "max_depth" : ("unlimited" if max_depth_value is None else max_depth_value),
    "min_samples_leaf" : chosen_params["min_samples_leaf"],
    "threshold" : chosen_threshold,
    "chosen_from" : chosen_from
}])
final_config_df.to_csv(OUTPUT_DIR / "part_c_final_config.csv", index=False) 

print("\nFinal configuration:")
print(final_config_df.to_string(index=False))
