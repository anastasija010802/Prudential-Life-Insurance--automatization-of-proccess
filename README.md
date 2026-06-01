# Prudential Life Insurance Risk Prediction


## Resources

-- Dataset: https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data

-- Project report: https://github.com/anastasija010802/Prudential-Life-Insurance--automatization-of-proccess/blob/main/dokumentacija/ML_projekat_dokumentacija.docx


## Project Overview

This project focuses on predicting life insurance risk categories based on people's habits, medical informations, genetics, physical characteristics etc. The dataset contains anonymized insurance application data, and the goal is to build a machine learning model that can classify applicants into appropriate risk response levels.The main challenge of this problem is that the target variable contains multiple ordered classes, and the class distribution is imbalanced. Because of that, the project includes data preprocessing, handling missing values, model comparison, hyperparameter tuning and evaluation using metrics suitable for multiclass classification.


## Problem Statement

The task is a supervised machine learning classification problem.  
The target variable is 'Response", which represents the risk category assigned to an insurance applicant.

The objective is to train and evaluate different machine learning models and select the model that performs best on unseen validation data.


## Dataset description

The main dataset, train.csv, contains 128 variables, including the target variable Response. The dataset consists of preprocessed and encoded features representing applicants' demographic, medical, insurance, and lifestyle information.

Several types of variables are present in the dataset:

Numerical variables representing continuous measurements.
Categorical variables encoded as numeric values.
Discrete variables, which are treated as categorical features due to their nature and interpretation.
Medical Keyword variables, a set of binary indicators (0/1) representing the presence or absence of specific medical conditions or keywords. Each observation may contain multiple active medical keywords.

The target variable, Response, represents the risk level assigned to an insurance applicant and serves as the prediction target for the multiclass classification task.

The dataset contains missing values in several features, making data preprocessing and imputation important steps in the modeling process.


## Data preprocessing

Several steps are applied before diving into machine learning models and problem solving. 

-- Feature transformations(changing type of features into appropriate)
-- Droping some columns (Nan above 75%)
-- Train and test split before any preprocessing steps in order to prevent data leakage and ensure realistic model evaluation
-- Missing value imputation
-- Reducing the number of categories by grouping infrequent categories into an "Other" category


## Model Development

Several machine learning algorithms were evaluated to predict the insurance risk response:

-- Logistic Regression
-- Decision Tree
-- Random Forest
-- XGBoost

All models were trained using the same training dataset to ensure a fair comparison.


## Hyperparameter Tuning

Hyperparameters were optimized using GridSearchCV with stratified cross-validation. The primary optimization metric was Cohen's Kappa score due to the ordinal and imbalanced nature of the target variable.


## Model Evaluation

Models were evaluated using several performance metrics:
Quadratic Kappa
AUC
Accuracy

Confusion matrices were also analyzed to better understand classification performance across response classes.

| Model | Kappa | AUC | Accuracy | Weighted F1-Score |
|--------|--------|----------|----------|----------|
| Logistic Regression | 0.5 | 0.79 | 0.43 | 0.44 |
| Decision tree | 0.42 | 0.78 | 0.43 | 0.43 |
| Random Forest | 0.47 | 0.82 | 0.52 | 0.45 |
| XGBoost | 0.52 | 0.83 | 0.49 | 0.5 |


## Conclusion 

XGBoost achieved the best overall performance among all evaluated models. Although Random Forest obtained the highest accuracy, XGBoost produced the highest Cohen's Kappa score, AUC, and Weighted F1-score. Since the target variable was imbalanced, model selection was primarily based on Cohen's Kappa rather than accuracy.


## Future improvments

Possible directions for future work include:

-- Exploring additional feature engineering techniques
-- Testing ensemble and stacking approaches
-- Applying advanced methods for handling class imbalance
-- Performing more extensive hyperparameter optimization
-- Investigating feature selection techniques to reduce noise and improve model interpretability