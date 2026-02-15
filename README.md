# ML_Assignment

1️. Problem Statement

The objective of this project is to build multiple machine learning classification models to predict whether an individual earns more than $50K per year based on demographic and employment-related attributes.

This is a binary classification problem where:

Class 0 → Income ≤ 50K

Class 1 → Income > 50K

The goal is to compare different machine learning algorithms and evaluate their performance using multiple evaluation metrics.

2️. Dataset Description

The dataset used is the Adult Income Dataset from the UCI Machine Learning Repository.

Dataset Details:

Number of Instances: ~48,000

Number of Features: 14

Target Variable: income

Type: Binary Classification

Features Include:

Age

Workclass

Education

Marital Status

Occupation

Relationship

Race

Sex

Capital Gain

Capital Loss

Hours per week

Native Country

Preprocessing Steps:

Missing values replaced with NaN and removed

Target variable encoded (≤50K = 0, >50K = 1)

One-hot encoding applied

Feature scaling using StandardScaler

Train-test split (80:20)

3. Models Used

The following six classification models were implemented:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (GaussianNB)

Random Forest (Ensemble)

XGBoost (Ensemble)

4️. Evaluation Metrics

Each model was evaluated using:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

5. Model Comparison Table

| ML Model            | Accuracy             | AUC                 | Precision                | Recall                | F1 Score                | MCC                  |
| ------------------- | ----------------     | -----               | ---------                | ------                | --------                | -----                |
| Logistic Regression | (0.851077943615257) | (0.9037746884712360) | (0.7393736017897090)     | (0.6000907852927830)  | (0.6624906038586820)    | (0.5733679846945860) |
| Decision Tree       | (0.8135986733001660) | (0.7463410886512680) | (0.6179826563213140)     | (0.6146164321379940)  | (0.6162949476558940)    | (0.49319773348308200) |
| KNN                 | (0.8220011055831950) | (0.8441679472187420) | (0.6548302872062660)     | (0.5692237857467090)  | (0.6090335114133070)    | (0.4965657595718530) |
| Naive Bayes         | (0.5659480375898290) | (0.806168324584092) | (0.3532617952648610)     | (0.9414434861552430)  | (0.5137478325489220)    | (0.34759713567935200) |
| Random Forest       | (0.8541735765616360) | (0.9027085384748790) | (0.7376344086021510)     | (0.6227871084884250)  | (0.9027085384748790)    | (0.5856595810430940) |
| XGBoost             | (0.8759535655058040) | (0.9296058044735310) | (0.9296058044735310)     | (0.657739446209714)  | (0.7208955223880600)    | (0.6470176266338710) |

6. Observations on Model Performance
   
ML Model	                               Observation
---------------------------    -----------------------------------------------------------------------------
Logistic Regression	           Performs well as a baseline model. Provides stable performance with good generalization.
Decision Tree	               Can overfit on training data but captures non-linear relationships effectively.
KNN	                           Sensitive to scaling and value of K. Moderate performance on this dataset.
Naive Bayes	                   Assumes feature independence. Performs reasonably well but slightly lower compared to ensemble models.
Random Forest	               Improves performance over Decision Tree by reducing overfitting. Strong overall results.
XGBoost	                       Achieves highest performance due to boosting technique and better optimization.

7. Streamlit Application

The project includes a Streamlit web application with:

Dataset upload option (CSV)

Model selection dropdown

Evaluation metrics display

Confusion matrix visualization

8. Repository Structure

project-folder/
│-- app.py
│-- ml_assignment02.py
│-- requirements.txt
│-- README.md
└── models/


