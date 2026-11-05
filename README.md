DELHIVERY LAST-MILE DELAY INTELLIGENCE PIPELINE:- this is an AI/ML capstone project which is focused on analyzing shipment delay , understanding operational factor
which influence delivery performance and preparing data for shipment delay prediction.

PROJECT OVERVIEW:- Last-mile delhivery performance can be influenced by following factor: 
1) delivery distance
2) service type
3) hub capacity
4) operational condition
here we used shipment data to build a data processing and analysis pipeline. part A focus on data cleaning , feature engineering, EDA and identifying patterns associated with delayed deliveries. 

OBJECTIVE:- 1) clean and validate shipment data.
            2) remove duplicate record and handle missing values
            3) standardize service type and payment information.
            4) merge shipment record with hub information.
            5) analyze delay by service type, hub and delivery distance
            6) find correlation between delivery duration and numerical variables.
            7) prepare modelling table for furthur machine learning operations.

PROJECT STRUCTURE:- 1) data:- 
                              i)  hubs.csv
                              ii) raw_shipment.csv
                              iii) new_shipment.csv 
                              iv) modelling_table.csv 
                    2) outputs:- 
                                i) part_a_checks.csv
                                ii) part_a_correlations.csv
                                iii) part_a_delay_by_group.csv
                    3) src:- 
                             i) shipment_prep.py
                    4) generate_data.py
                    5) part_a_clean_eda.py
                    6) requirements.txt
                    7) README.md

DATASETS:- 1) data/raw_shipments.csv :- contain original shipment records 
           2) data/hubs.csv :- contain hub information and operational attributes 
           3) data/new_shipments.csv :- contain additional shipment records 
           4) data/modelling_table.csv :- cleaned dataset with engineered features and delay labels 

PART A: DATA CLEANING AND EDA 
        1) data cleaning:- the preprocessing pipeline contain:- 
                           i) remove duplicate records .
                           ii) standardize service type
                           iii) replacing missing payment with unknown 
                           iv) convert declared value into numeric INR values , remove unnecessary comma
                           v) standardize dispatch time 
                           vi) treat parcel weight > 100kg as invalid values 
                           vii) enter median values in place of missing numerical values v
                           viii) merge hub data with shipment data ]
        2) feature engineering:- pipeline creates following features:- 
                            i) promised_hours(delivery time commitment by the company)
                            ii) is_cod ( indicate whether payment is in cod) 
                            iii) is_weekend ( indicate if dispatch occured on weekend) 
                            iv) dispatch_hour (indicate dispatch hour)
                            v) log_declared_value ( log transformend declared value inorder to overcome the outliers) 
                            vi) is_delayed ( indicate whether actual delivery time is more than the promised hours by the company)

        3) promised delivery hours :- express = 36
                                      standard = 60
                                      economy = 96 

        4) the analysis produced the following result:- 
                                                      i) raw records = 1224
                                                      ii) removed duplicate record = 24
                                                      iii) records after deduction = 1200
                                                      iv) delayed shipments = 278 
                                                      v) overall delay rate = 0.231667

        5) delay by service type :- i) express:- 119 shipments delayed out of 243. delay rate = 48.97%
                                    ii) standard :- 132 delayed shipments out of 642 .delay rate = 20.56% 
                                    iii) economy :- 27 delayed shipments out of 315. delay rate = 8.57% 

        6) delay rate by hub tier :- i) tier1 :- 16 delayed shipment out of 400. delay rate = 4% 
                                     ii) tier2 :- 86 delayed shipment out of 400. delay rate = 21.50%
                                     iii) tier3 :- 176 delayed shipment out of 400. delay rate = 44% 

        7) delay rate by distance :- i) 0-200km :- 6 out of 274. delay rate = 2.19%
                                     ii) 201-400km :- 77 out of 573. delay rate = 134.68%
                                     iii) 401-600km :- 120 out of 275. delay rate = 43.64% 
                                     iv) >600km :- 75 out of 88. delay rate = 85.23%

        8) correlation analysis :- we find the correlation with respect to delivery hours.
                                    i) promised_hours:- correlation with delivery hours is 0.8372
                                    ii) distance_km:- correlation with delivery_hours is 0.4013
                                    iii) n_stops:- correlation with delivery_hours is 0.3247 

        9) declared shipment vsalue is log transformed because we want to rduce the influence of large values so that we can easily do the analysis.

        10) output files:- i) outputs/part_a_checks.csv :- data quality checks and summary statistics 
                          ii) outputs/part_a_delay_by_group.csv :- delay analysis by service type, hub and distance 
                          iii) outputs/part_a_correlations.csv :- contain correlation output of promised_hours, distance_km, n_stops with delivery_hours. 

PART B (DELIVERY TIME PREDICTION) :- 
                                     In the part B of capstone project, our objective is to predict the expected delivery time of shipments in hours using machine learning regression algorithms. 

1) dataset and data splitting:-
i) total cleaned rows of shipments:-1200
ii) training rows:- 840
iii) validation set :- 180 
iv) testing rows :- 180
v) target variable :- delivery_hours 
vi) input features :- 17 predefined features with exact orders 
vi) here the dataset was split using random_state=42 

2) model applied :-
i) mean baseline regressor 
ii) decision tree regressor 
iii) linear regression 
iv) random forest regressor
v) gradient boosting regressor 
                            
a) Baseline model result:- 
the baseline model will predict the mean delivery time for every shipment. 
r2_score of baseline model is :- -0.0212 
here the -ve sign means that the baseline predictions perform worse than predicting the actual mean delivery time as refrence. therefore we need a more effective model. 
b) Decision tree depth analysis:- in this model we used different max_depth values for testing and understanding the effect of model complexity.
tree depth experiment:
max_depth  train_rmse  val_rmse
        1   11.633436   12.132363
        2    9.368688   10.411443
        3    7.529527   8.703332
        4    6.099872   7.389621
        6    4.711358   6.279637
        8    3.336034   6.790075
       12   1.361281   7.353997
unlimited   0.000000   7.836781

Selected decision tree depth: 6
here we can see that depth 1 resulted in underfitting .here the tree was too simple to capture the relationship of data. unlimited depth tree achived zero training rmse but higher validation rmse this indicate that the model with unlimited depth is memorizing the training data instead of generalizing.

c) validation model comparison:- Validation ranking - lowest RMSE is best:
1. linear_regression: RMSE = 4.8912hours
2. gradient_boosting: RMSE = 4.9165hours
3. random_forest: RMSE = 5.5095hours
4. decision_tree: RMSE = 6.2796hours
here we can clearly observe that linear regression has lowest validation rmse.

d) gradient boosting learning rate:- Gradient boosting learning rate experiment:
learning_rate  val_rmse   val_r2
         0.01 10.001759 0.712652
         0.10  4.916525 0.930566
         1.00  7.370066 0.843974
here we can observe that learning rate of 0.01 resulted in underfitting because the model learned too slowly with 100 estimators.
the learning rate of 1.0 clearly indicating the aggressive behaviour which is reducing the generalization performance.
the learning rate of 0.1 provide the balanced output between learning and generalization. 

e) cross validation result:- Five fold cross validation:
              model  mean_r2   std_r2
  linear_regression 0.926133 0.004754
  gradient_boosting 0.935266 0.004477 
here we can clearly observe that the gradient boosting model achived higher mean cross validation r2 score than linear regression model.however, the linear regression model achived a slightly lower validation rmse and also it is a very simpler and easily interpretable model. therefore we will select linear regression model in final test evaluation. 

f) final test result:- All model metrics:
            model split       MAE        MSE      RMSE        R2
    mean_baseline   val 15.795189 355.520424 18.855249 -0.021221
linear_regression   val  3.922162  23.923532  4.891169  0.93128
    decision_tree   val  5.097064  39.433846  6.279637  0.886728
    random_forest   val  4.288197  30.355005  5.509538  0.912806
gradient_boosting   val  3.916577  24.172216  4.916525  0.930566
linear_regression  test  3.567078  20.462444  4.523543  0.949250 
here the linear regression model was selected as the final model because it achieved the lowest validation rmse among all evaluated models.although,gradient boosting achieved va higher cross validation r2 score but linear regression model shows the better performance in test dataset.here the final model will provide a reliable foundation for predicting the shipment delivery times and also it will support the furthur classification and operational analysis.

g) output files :- 
i) part_b_checks.csv :- contains the data and split validation checks. 
ii) part_b_tree_depth.csv :- contains the decision tree comparsion
iii) part_b_metrics.csv :- it contain the regression model performance.
iv) part_b_gb_learning_rate.csv :- it contain the gradient boosting learning rate comparison. 
v) part_b_cv.csv :- it contains the cross validation results. 

PART C (SHIPMENT DELAY CLASSIFICATION) :- 
                                        the objective of part c is to classify shipmenmts as delayed or non-delayed using machine learning classification          algorithms.The targwet variable is is_delayed:- i) 1 ---> delayed shipments
                                                ii) 0 ----> non delayed shipments
                                                
1) dataset and splitting :-
i) total shipments :- 1200
ii) training set rows :- 840
iii) validation set rows :- 180
iv) test set rows :- 180
v) delayed shipments in training sets :- 195 
vi) delayed shipments in validation sets :- 42 
vii) delayed shipments in test set :- 41 
viii) random_state = 42 
ix) splitting method :- stratified train-validation-test split

2) models implemented :- the following models were evaluated. 
i) blind baseline models 
ii) logistic regression model 
iii) decision tree classifier
iv) random forest classifier 
a) initial model comparison :- Classification model evaluation:
               model  TN  FP  FN  TP  accuracy  precision   recall       f1
    blind_all_on_time 138   0  42   0  0.766667   0.000000 0.000000 0.000000
  logistic_regression 128  10   7  35  0.905556   0.777778 0.833333 0.804598
decision_tree_depth_6 128  10  11  31  0.883333   0.756098 0.738095 0.746988
random_forest_default 132   6  11  31  0.905556   0.837838 0.738095 0.784810
here we can clearly observe that the baseline model achieved the accuracy of 76.67% by predicting all shipments as non_delayed.but it still fails to identify the delayed shipments.Logistic regression model achieved the highest F1 score and random forest model achieved the highest precision. 

b) feature selection :- here we perofrm the feature selection to identify the most useful predictors of shipment delays.
 Features score:
        feature    f_score  f_rank  rf_importance  is_noise
       distance_km 389.213126       1       0.270831     False
          hub_tier 155.348313       2       0.039929     False
hub_daily_capacity 144.134684       3       0.081754     False
           n_stops 125.479792       4       0.069879     False
    promised_hours  84.417725       5       0.087394     False
         rain_flag  53.032064       6       0.045019     False
   sale_event_flag  36.728141       7       0.033463     False
 label_print_batch   4.584717       8       0.045985      True
            is_cod   3.894937       9       0.010593     False
        is_weekend   2.693921      10       0.008696     False
     dispatch_hour   2.447188      11       0.039884     False
      tape_roll_no   1.905205      12       0.047567      True
log_declared_value   1.370792      13       0.045951     False
sticker_serial_mod   1.277617      14       0.049857      True
  parcel_weight_kg   0.555515      15       0.042229     False
     shelf_slot_no   0.139270      16       0.043562      True
  scanner_build_no   0.022956      17       0.037407      True       
  Selected features:
  ['promised_hours', 'distance_km', 'n_stops', 'rain_flag', 'sale_event_flag', 'hub_tier', 'hub_daily_capacity']
                 
Combined noise-column forest importance share: 0.2244. This indicate that approximately 22.44% of the feature importance was attributed to noise columns in the evaluation analysis. 

c) validation result :-  
Validation threshold experiment:
threshold  TN  FP  FN  TP  accuracy  precision   recall       f1
      0.1  94  44   1  41  0.750000   0.482353 0.976190 0.645669
      0.2 109  29   3  39  0.822222   0.573529 0.928571 0.709091
      0.3 121  17   3  39  0.888889   0.696429 0.928571 0.795918
      0.4 125  13   5  37  0.900000   0.740000 0.880952 0.804348
      0.5 129   9   7  35  0.911111   0.795455 0.833333 0.813953
      0.6 134   4  10  32  0.922222   0.888889 0.761905 0.820513
      0.7 134   4  15  27  0.894444   0.870968 0.642857 0.739726
      0.8 134   4  23  19  0.850000   0.826087 0.452381 0.584615
      0.9 136   2  30  12  0.822222   0.857143 0.285714 0.428571         
here we can clearly observe that threshold 0.6 has highest accuracy, precision and f1 score. threshold 0.5 have highest recall.we can conclude that 0.6 will be a better threshold but question asked us to use 0.5 threshold so we used that.
     
d) final model configuration :- 
Final configuration:
selector_k  n_estimators max_depth  min_samples_leaf  threshold                    chosen_from
         7           100 unlimited                 1        0.5 default_forest_same_7_features

e) final test result :-   final test matrics:
  threshold  TN  FP  FN  TP  accuracy  precision   recall      f1
        0.5 125  14  10  31  0.866667   0.688889 0.756098 0.72093
  the model correctly identifies 31 delayed shipments out of 41 actual delayed shipments in the test data.

f) conclusion:- the final random forest classifier model uses 7 selected features and probability threshold of 0.5.the model achieve a test accuracy of 86.67%, precision 68.89%, recall 75.61%, and f1 score of 72.09%. 

PART D (HUB SEGMENT AND MODEL DEPLOYMENT) :- 
                                            the objective of part D is to segment delivery hubs based on their operational characteristics, package the final delay classification model, and generate actionable predictions for new shipments. 

a) hub profile :- we created the hub profile .
  hub_id  shipments  mean_delivery_hours  delay_rate  mean_distance_km  
0  HUB01         50              45.6850        0.02           185.061   
1  HUB02         50              47.7904        0.04           195.130   
2  HUB03         50              49.9894        0.04           200.033   
3  HUB04         50              44.9116        0.04           191.363   
4  HUB05         50              45.6334        0.00           184.236

b) hub segment profile:- 
Hub segments:
   hub_id   segment     hub_city  hub_tier
0   HUB01  Standard       Mumbai         1
1   HUB02  Standard        Delhi         1
2   HUB03  Standard    Bengaluru         1
3   HUB04  Standard    Hyderabad         1
4   HUB05  Standard      Chennai         1
5   HUB06  Standard         Pune         1
6   HUB07     Watch      Kolkata         1
7   HUB08  Standard    Ahmedabad         1
8   HUB09     Watch       Jaipur         2
9   HUB10     Watch      Lucknow         2
10  HUB11     Watch   Chandigarh         2
11  HUB12     Watch       Indore         2
12  HUB13  Escalate       Nagpur         2
13  HUB14     Watch   Coimbatore         2
14  HUB15  Escalate        Kochi         2
15  HUB16     Watch        Surat         2
16  HUB17  Escalate        Patna         3
17  HUB18  Escalate       Ranchi         3
18  HUB19  Escalate     Guwahati         3
19  HUB20  Escalate       Raipur         3
20  HUB21  Escalate  Bhubaneswar         3
21  HUB22     Watch     Dehradun         3
22  HUB23  Escalate     Siliguri         3
23  HUB24  Escalate     Agartala         3

c) hub clustering analysis :- first we tried different clustering configurations and evaluated them : 
Separation result:
   K  across_over_within cluster_sizes
0  2            2.893964          15-9
1  3            5.392845         9-8-7
2  4            6.304705       8-7-5-4
3  5            6.886628     8-7-4-3-2
here the separation ratio increased with the number of clusters. the 3 cluster configuration was used for the operational segment into standard, watch and escalate category. 

d) hub operational segmentation :- 
Segment profiles:
    segment  hubs  mean_delivery_hours  delay_rate  mean_distance_km   mean_n_stops
0  Standard     7            47.239629    0.025714        188.929571   55.368571  
1     Watch     8            56.813675    0.155000        314.176250   70.717500
2  Escalate     9            64.650444    0.460000        470.475667   84.344444  

interpretation of segment :- 
i) Standard :- it represnt relatively lower delivery time and delay rate accompanied by shorter average shipment distance and low operational workload.
ii) Watch :- it represnt moderate delivery time and delay rate.it requires regular monitoring of operational performance. 
iii) Escalate :- highest average delivery time and delay rate accompanied by longr average distance and higher operational workload. it requires large operational attention.

e) hub-tier and segment mismatch analysis :- these are mismatched hubs which requires additional investigation.
Tier vs segment mismatch:
   hub_id   segment  hub_city  hub_tier    assigned segment
6   HUB07     Watch   Kolkata         1               Watch
12  HUB13  Escalate    Nagpur         2            Escalate
14  HUB15  Escalate     Kochi         2            Escalate
21  HUB22     Watch  Dehradun         3               Watch 

f) model packaging and reload verification :- we packaged our model and relaoded it and verified it against original model .
i) delay_model.pkl
ii) delay_threshold.csv
iii) feature_order.csv 
iv) impute_values.csv
the maximum absolute probability difference after reloading is 0.0. this conforms that the packaged model reproduces the original model's predicted probabilities during the reload. 

g) new shipment prediction result :- now we applied the reloaded model on the new shipments and combine them with their hub segment and did the following predictions :
Number of shipments: 48
Number of flagged shipments: 8
Action recommended:
action
No action                  40
Escalate to hub manager     6
Proactive customer call     2
Name: count, dtype: int64

h) action logic :- 
i) flagged shipment assigned to escalated segment which will together recommened for escalation to hub manager. 
ii) other flagged shipments are recommended for proactive customer calls .
iii) non flagged shipments will receive no action .

i) output files :- 
1) part_d_separation :- represent clustering segment analysis.
2) hub_profile.csv :- contain operational hub profile.
3) hub_segments.csv :- contain hub-to-segment mapping
4) segment_profile.csv :- contain hubs summaries
5) tier_vs_segment_mismatch.csv :- hub-tier mismatch analysis
6) part_d_reloaded_checks.csv :- model reload verification
7) new_shipment_actions.csv :- it represent the new shipment actions and recommended actions . 
