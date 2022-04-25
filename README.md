
An analysis using Machine Learning algorithms to identify credit card risk using a dataset from LendingClub.

# Overview

The purpose of this analysis is to understand how to utilize `Machine Learning` statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on **Supervised Learning** using a free dataset from **LendingClub** This is called **"Supervised Learning"** is because the data includes a labeled outcome. 

To complete this analysis, we use different `Machine Learning` techniques to train and evaluate the data with unbalanced classes. The dataset from the **LendingClub** has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, we needed to employ various `Machine Learning` algorithms to resample the data. These algorithms include `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, `SMOTEENN`, `BalancedRandomForestClassifier`, and `EasyEnsembleClassifier`.

# Results

As mentioned in the overview, we use `Machine Learning` to resample the dataset using `Python` libraries: `scikit-learn` and `imbalanced-learn` evaluate the results and provide a comparison for our analysis. 

The original dataset loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk". 


Using built in train and test method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.   

## Deliverable 1: Use Resampling Models to Predict Credit Risk

### Oversampling

**`RandomOverSampler Model`** randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.

  * Balanced accuracy score: 66%.

  * The "High Risk" precision rate was only 1% with the recall at 69% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and recall at 63%.  
  
  ![oversample_result](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/RandomOversamplingResult.png)
  

**`SMOTE (Synthetic Minority Oversampling Technique) Model`**, like `RandomOverSampler` increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. 

  * The balanced accuracy score 65.1% almost similar to RandomOversampling.

  * Like `RandomOverSampler`, the "High Risk" precision rate again was only 1% with the recall degraded to 62% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and an improved recall at 68%.  

  
  ![smote_result](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/SMOTEOversamplingResult.png)

### Undersampling

**`ClusterCentroids Model`**, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

  * Balanced accuracy score was lower than the oversampling models at 54.7%.


  * The "High Risk" precision rate again was only at 1% with the recall at 68% giving this model an F1 score of 1%.
  * "Low Risk" had a precision rate of 100% and with a lower recall at 41% compared to the oversampling models.  

  ![under_sampling_result](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/UnderSampledResult.png)

## Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

### Combination Sampling

**`SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model`** combines aspects of both oversampling and undersampling. The model classified 51,358 records as High Risk and 46,652 as Low Risk.

  * The balanced accuracy score improved to 65.8% when using a combined sampling model.

  * The "High Risk" precision rate did not improve was only 1%, however the recall increased to 73% giving this model an F1 score of 2%.
  * "Low Risk" still showed a precision rate of 100% with the recall at 58%.  
  

  ![SMOTEENN_result](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/SMOTEENNCombinedResult.png)

## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Compare two new `Machine Learning` models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.


**`BalancedRandomForestClassifier Model`**, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class. 

  * The balanced accuracy score increased to 83.2% for this model.

  * The "High Risk precision rate increased to 4% with the recall at 77% giving this model an F1 score of 7%.
  * "Low Risk" still had a precision rate of 100% with the recall at 89%.  
  * The top feature by importance was "total_rec_prncp" at 7.9% of the total.
)
  
  ![balance_random_result](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/BalancedRandomForest_Result.png)

  ![balance_random_feature](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/BalanceRandomFeatureResult.png) 

**`EasyEnsembleClassifier Model`**, a set of classifiers where individual decisions are combined to classify new examples.

  * The balanced accuracy score increased to 90.1% with this model.


  * The "High Risk precision rate increased to 8% with the recall at 86% giving this model an F1 score of 14%.
  * "Low Risk" still had a precision rate of 100% with the recall now at 95%.  

  
  ![easy_ensemble_result](https://github.com/vijaycse/Credit_Risk_Analysis/blob/master/images/EasyEnsembleResult.png)

# Summary

In reviewing all six models, the `EasyEnsembleClassifer` model yielded the best results with an accuracy rate of 90.1% and a 8% precision rate when predicting "High Risk candidates. The sensitivity rate (aka recall) was also the highest at 86% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 95% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

**Ranking of models in descending order based on "High Risk" results:**
* `EasyEnsembleClassifer`: 90% accuracy, 9% precision, 95% recall, and 97% F1 Score
* `BalancedRandomForestClassifer`: 83.2% accuracy, 4% precision, 89% recall and 94% F1 Score
* `SMOTE`: 65.2% accuracy, 1% precision, 68% recall and 80% F1 Score
* `SMOTEENN`: 65.2% accuracy, 1% precision, 59% recall and 73% F1 Score
* `RandomOverSampler`: 66.3% accuracy, 1% precision, 63% recall and 77% F1 Score
* `ClusterCentroids`: 54.7% accuracy, 1% precision, 41% recall and 58% F1 Score

A side note that should be considered is that original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew the results greatly as there is a risk that the `Machine Learning` algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. This margin of risk might not be something that banks would be comfortable accepting.

# Resources

* Dataset from LendingClub: [LoanStats_2019Q1](https://raw.githubusercontent.com/vijaycse/Credit_Risk_Analysis/master/resources/LoanStats_2019Q1.csv)
* Software: Python 3.8.5, Anaconda 4.9.2 and Jupyter Notebooks 6.1.4
