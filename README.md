
![image](https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/3079e6b1-6974-4591-ba37-374f493b3cf2)



# Sales Time Series Forecasting Using Machine Learning Techniques (Random Forest, XGBoost, Stacked Ensemble Regressor)

Developed as a course project for the program "Business Analytics: Operational Research and Risk Analysis" at the Alliance Manchester Business School.

This code can be viewed through jupyter nbviewer via this <a href="https://nbviewer.org/github/DSkapinakis/sales-time-series-forecasting-ml-models/blob/main/Sales_forecast_ml.ipynb">link</a>

# Project Overview

The objective of this project is to build a predictive model to forecast 6 weeks of daily sales for 1,115 drug stores in Europe. 

Key steps of the project:
1. Exploratory Data Analysis (EDA)
2. Datetime Objects preprocessing
3. Time Series K-Means clustering using Dynamic Time Warping (to effectively capture curve similarity across time)
4. Generic Preprocessing and Feature Engineering
5. Cluster-specific EDA
6. Variance Inflation Factor (VIF) Backwards Feature Selection (per cluster)
7. Development of Naive Models based on historical sales data (day of week, day of month, day of year)
8. Introduction of historical sales proxy features (Weekly, Monthly based)
9. Three sets of ML models were developed per cluster (No proxy, weekly proxy, monthly proxy)
10. Visualizations of sales predictions for randomly selected stores of each cluster

The ML models used are:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Ensembled Stacking model using Random Forest and XGBoost as weak learners and Linear Regression ad Meta Learner

The models were evaluated based on the the Root Mean Square Percentage Error (RMSPE) and R-squared metrics for the training and validation sets. However, RMSPE was primarily used to measure the performance of each model since R-squared might not be suitable for capturing the goodness of the fit of non-linear relationships.

<div align="center"> <img width="176" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/b1dd85a2-4938-452a-bf41-b7260fd9fca5"></div>

*<h6> N is the total number of data records for accuracy measurement, yi is the actual sales for the ith record, ŷi is the sales forecast for the ith record. Zero sales were excluded from the calculation*


# Installation and Setup

## Codes and Resources Used
- **Editor Used:**  JupyterLab
- **Python Version:** Python 3.10.9

## Python Packages Used

- **General Purpose:** `statsmodels, scipy, time, math`
- **Data Manipulation:** `pandas, numpy` 
- **Data Visualization:** `seaborn, matplotlib`
- **Machine Learning:** `scikit-learn, tslearn`

# Data

- `stores.csv`: contains supplementary information for the 1,115 drug stores (1115 rows, 10 columns)
  
  <img width="452" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/c68b844b-41c0-4ab7-9523-3f7cb9f86ce8">

- `train.csv`: contains the historical sales data, which covers sales from 01/01/2013 to 31/07/2015 (1017209 rows, 9 columns)

  <img width="451" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/f0f4473f-e1dd-41d7-b6c8-50eaade02eb0">

- `test.csv`: identical to train.csv, except that Sales and Customers are unknown for the period of 01/08/2015 to 17/09/2015 (41088 rows, 9 columns)

# Methodology

The overall project design was based on the Cross Industry Standard Process for Data Mining (CRISP-DM), which consists of Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation and Deployment

## Figure 1

<img width="1269" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/20e818ad-85be-4acb-9399-759577ebeed4">

The data pre-processing pipeline is illustrated in **Figure 1**. Initially `store_train` was split into training and validation data sets. A preliminary round of data transformation and feature engineering based on the `store_train` set has been applied across all `store_train`, `store_validate` and `store_test`. 
With the pre-processed dataset in hand, clustering was performed to separate the dataset into several clusters for model partitioning. Given the cyclical nature of the sales, Time Series K-Means clustering was employed using Dynamic Time Warping (DTW) as the distance metric to effectively capture curve similarity across time. To remove the effects of differences in the magnitude of sales across stores, the store-specific historical sales data was first transformed by applying *TimeSeriesScalerMeanVariance()* to standardize sales on a store-by-store basis, resulting in each store's sales having zero mean and unit variance. 
Additional cluster-specific EDA was then performed on the clustered dataset, aiding in handling missing values and making data feature engineering decisions on each individual cluster. The final clustered training datasets were checked for multicollinearity using the variance inflation factor index (VIF). Features with the highest VIF were iteratively removed until all features had a value less than five. This resulted in the final training, validation, and testing datasets for each cluster.

## Figure 2

<img width="1000" height = "350" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/bb66b420-f64c-4da5-abc7-257fc8bce635">

The modelling pipeline is demonstrated in **Figure 2**, with a separate set of models being developed for each cluster. Naïve models were first developed as a benchmark for the ML models. The prediction from the naïve models for open stores was either based on the historical weekly average or the monthly average of that specific store, while the prediction for closed stores was 0. Upon retrieving the benchmark results, five selected ML models were developed with and without the *sales_proxy* from the naïve models and further validation was performed. Stacking was then applied on the most robust models to address any potential overfitting issues. To prevent potential data leakage from the *sales_proxy* and one-hot-encoded (day-of-month) variables, validation was exclusively conducted on the validation set, as opposed to adopting a cross-validation approach. Finally, permutation importance method was used to extract the feature importance of the final models and provide business recommendations.

# Results and evaluation

The clustering analysis resulted in 4 clusters of stores (A, B, C, D), and the final features used for each cluster-specific model after VIF backward selection are demonstrated below:

<div align="center"> <img width="682" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/d33084e1-dcc8-4124-9147-108cb3756c69"></div>


## Final Cluster-Specific Models

The basic ML models without the *sales_proxy* dominated the naïve ones, proving that ML is an effective technique. However, the more advanced ML models with the *sales_proxy* feature showed a better behaviour, and the *monthly_proxy* ones demonstrated the lowest RMSPE scores without overfitting. For the *monthly_proxy* models, Random Forest and XGBoost appeared to be the most powerful, thus they were also stacked as weak learners with the Ensemble Stacking Method (Meta Learner: Linear Regression). The results prove that stacking was successful since in most of the clusters the difference between training and validation error decreased, thus reducing overfitting. The selected models with the highest performance for each cluster are depicted in the following table: 

<div align="center"> <img width="531" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/57f3408c-399e-425d-a75b-d622c0a2e155"></div>


## Permutation Feature Importance for final models across different clusters

<div align="center"> <img width="976" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/27ce3d7c-dc1e-4634-bb25-28776369f909"></div>
<br>
<div align="center"> <img width="747" alt="image" src="https://github.com/DSkapinakis/sales-time-series-forecasting-ml-models/assets/136902596/12741012-5bed-4f1d-9e0b-45a92631095b"></div>

The tables above illustrate the permutation feature importance across the different clusters. First, the model illustrates a strong seasonality pattern. The monthly *sales_proxy* successfully captured a significant amount of the monthly patterns and is the most important feature in terms of permutation importance. The day of the week (dow) was also a significant indicator in predicting sales. A trend in the historical data was that sales peaked in December, on Mondays and Sundays, and additionally at the beginning, middle, and end of each month. It was also discovered that school holidays impact sales more than state holidays. Other factors that influenced sales were promotions, with individual store promotions (Promo) appearing to be more effective than coupon-based mailing campaigns (Promo2). The distance to competitors (CompetitionDistance) also showcased some significance within clusters A and C.


# Assumptions, limitations, future improvements

The predictive models were based on some assumptions: 

- Monday and first day of month were assumed when computing Promo2Duration and CompetitiveLength
- Seasonality in sales led to the establishment of time information features
- Missing values in Open were imputed based on mode of other stores on the given date within the cluster

The last assumption covers a negligible limitation of the model, where it might not be accurate when predicting sales on days where stores are closed, because the model will not predict exactly zero sales. This should be unproblematic as it is more valuable to predict the sales when stores are open.
For future improvements, multiple things can be investigated; firstly, investigate underlying patterns of when stores are open to impute any missing records in a more pragmatic way. Secondly, a more systematic approach can be used to evaluate the clustering results. Thirdly, a model could be developed which uses cross validation with hyperparameter tuning instead of *sales_proxy* features, to compare results. Lastly, due to the fact that the sales proxy may bring multicollinearity issue to the model, the interpretability of the model could be trimmed. With more data available and the model being predictive enough, sales proxy could be removed in the future.





