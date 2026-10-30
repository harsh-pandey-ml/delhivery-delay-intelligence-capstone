DELHIVERY LAST-MILE DELAY INTELLIGENCE PIPELINE:- this is an AI/ML capstone project which is focused on analyzing shipment delay , understanding operational factor
which influence delivery performance and preparing data for shipment delay prediction.

PROJECT OVERVIEW:- Last-mile delhivery performance can be influenced by following factor: 1) delivery distance
                                                                                          2) service type
                                                                                          3) hub capacity
                                                                                          4) operational condition
here we used shipment data to build a data processing and analysis pipeline. part A focus on data cleaning , feature engineering, EDA and identifying patterns 
associated with delayed deliveries. 

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
