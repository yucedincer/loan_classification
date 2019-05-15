#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import os
from functools import reduce
import pandas_profiling

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, RobustScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsemble

import warnings;
warnings.filterwarnings('ignore');


# ### Importing the data

borrower = pd.read_csv("ds-borrower.csv", names = ['column'])
application = pd.read_table("ds-app.tsv", names = ['column'])
credit = pd.read_table("ds-credit.tsv", names = ['column'])
result = pd.read_table("ds-result.tsv", names = ['column'])

# Parsing the 'column' to multiple columns

borrower['column'].str.split().tolist()
borrower = pd.DataFrame(borrower['column'].str.split().tolist(), columns="CustomerID YearsAtCurrentEmployer YearsInCurrentResidence Age RentOrOwnHome TypeOfCurrentEmployment NumberOfDependantsIncludingSelf".split())
borrower = borrower.drop(borrower.index[0])
borrower = borrower[borrower['YearsAtCurrentEmployer'] != 'Mvd']
borrower['YearsAtCurrentEmployer'] = borrower['YearsAtCurrentEmployer'].replace('10+', '10')

application['column'].str.split().tolist()
application = pd.DataFrame(application['column'].str.split().tolist(), columns="CustomerID LoanPayoffPeriodInMonths LoanReason RequestedAmount InterestRate Co-Applicant".split())
application = application.drop(application.index[0])

credit['column'].str.split().tolist()
credit = pd.DataFrame(credit['column'].str.split().tolist(), columns="CustomerID CheckingAccountBalance DebtsPaid SavingsAccountBalance CurrentOpenLoanApplications".split())
credit = credit.drop(credit.index[0])

result['column'].str.split().tolist()
result = pd.DataFrame(result['column'].str.split().tolist(), columns="CustomerID WasTheLoanApproved".split())
result = result.drop(result.index[0])

'''
Creating a new feature 'interest' to calculate -
the interest payments to be collected
'''

application['interest'] = ((application['RequestedAmount'].astype(int) * 
                                        application['InterestRate'].astype(int)) / 100) * (application['LoanPayoffPeriodInMonths'].astype(int)/12)

# Changing the target variable to binary

result['WasTheLoanApproved'] = result['WasTheLoanApproved'].apply(lambda x: 1 if x == "Y" else 0)

# Merging the dataframes

data_frames = [borrower, application, credit, result]

df_merged = reduce(lambda x,y: pd.merge(x,y, on='CustomerID', 
                                        how='outer', ), data_frames)
df_merged = df_merged.drop_duplicates()
df_merged = df_merged.drop(df_merged.tail(2).index)

# Converting str values to float
float_cols = ['Age','NumberOfDependantsIncludingSelf', 
              'LoanPayoffPeriodInMonths', 'RequestedAmount', 
              'InterestRate','CurrentOpenLoanApplications',
             'YearsAtCurrentEmployer', 'YearsInCurrentResidence', 
              'interest']

for coll in float_cols:
    df_merged[coll] = df_merged[coll].astype(float)

# Creating the Categorical Values
categoricals = ['TypeOfCurrentEmployment',
                'LoanReason','Co-Applicant','CheckingAccountBalance',
                'DebtsPaid','SavingsAccountBalance', 'RentOrOwnHome',]

for col in categoricals:
    df_merged[col] = df_merged[col].astype('category')

# Dropping the Customer ID column.
df_merged = df_merged.drop('CustomerID', axis = 1)

# Removing rows with na values in the target column

df_merged = df_merged[~df_merged['WasTheLoanApproved'].isnull()]

# Generating dummy features for the categoricals

loan_features = df_merged.drop('WasTheLoanApproved',axis = 1) #c/out this
loan_approval = df_merged['WasTheLoanApproved']

df_merged = pd.get_dummies(loan_features, drop_first=True) #c/out this

df_merged = df_merged.reset_index(drop=True)

'''
Splitting the data, 90:10 split. 
The first tries were with the standard 80:20 split, 
but this is a small dataset and the 90:10 split provided better results.
'''

X_train, X_test, y_train, y_test = train_test_split(df_merged, loan_approval, 
                                                    test_size = 0.10, shuffle = True, 
                                                    random_state = 123)
'''
Impute the missing data using features means. 
Using Median or most_frequent instead of mean provides similar results.
'''

imp = Imputer(strategy = 'mean')
imp.fit(X_train)
X_train = imp.transform(X_train)
X_test = imp.transform(X_test)


# ## Balanced Random Forest

# from imblearn.ensemble import BalancedRandomForestClassifier
# from sklearn.metrics import balanced_accuracy_score
# from imblearn.metrics import geometric_mean_score

# # Ensemble classifier using samplers internally

# brf = BalancedRandomForestClassifier(n_estimators = 50, 
#                                      random_state=123,
#                                      n_jobs = -1, 
#                                      max_depth = 2, 
#                                      criterion = 'entropy')

# pip_baseline = make_pipeline(RobustScaler(), brf)
# scores = cross_val_score(pip_baseline,
#                          X_train, y_train,
#                          scoring = "roc_auc", cv = 5)


# brf.fit(X_train, y_train)
# y_pred_brf = brf.predict(X_test)

# XGBoost Model

# xgb = XGBClassifier(objective="binary:logistic",
#                     learning_rate=0.1,
#                     n_estimators=200,
#                     max_depth=2,
#                     subsample=0.7,
#                     random_state=123,
#                     colsample_bytree = 0.7,
#                     gamma = 0.1,
#                     min_child_weight = 6,
#                     reg_alpha = 0.01)

# pip_baseline = make_pipeline(RobustScaler(), xgb)
# scores_xgb = cross_val_score(pip_baseline,
#                          X_train, y_train,
#                          scoring = "roc_auc", cv = 5)

# '''
# RobustScaler: Standarization will be less influenced by the outliers, 
# i.e. more robust. It centers the data around the median and 
# scale it using interquartile range (IQR)
# '''

# # fit XGB to plot feature importances
# xgb.fit(X_train, y_train)

# xgb_predictions = xgb.predict(X_test)


## Logistic Regression

logistic = LogisticRegression(penalty = 'l2', 
                              C = 0.1)

pip_baseline = make_pipeline(RobustScaler(), logistic)
scores = cross_val_score(pip_baseline,
                         X_train, y_train,
                         scoring = "roc_auc", cv = 5)

logistic.fit(X_train,y_train)

logistic_predictions = logistic.predict(X_test)
logistic_prob = logistic.predict_proba(X_test)

scores_log = cross_val_score(logistic,
                         X_train, y_train,
                         scoring="roc_auc", cv=10)


# ### Stacking

# from vecstack import StackingTransformer

# estimators = [("brf", BalancedRandomForestClassifier(n_estimators = 50, 
#                                                      random_state=123,
#                                                      n_jobs = -1, 
#                                                      max_depth = 2)),
#               ("xgb", XGBClassifier(objective="binary:logistic",
#                     learning_rate=0.1,
#                     n_estimators=200,
#                     max_depth=2,
#                     subsample=0.7,
#                     random_state=123,
#                     colsample_bytree = 0.7,
#                     gamma = 0.1,
#                     min_child_weight = 6,
#                     reg_alpha = 0.01)),
#               ("logit", LogisticRegression(penalty = 'l2', 
#                                            C = 1, 
#                                            class_weight={1:4, 0:6}))   
# ]

# stack = StackingTransformer(estimators, regression = False, verbose = 2)

# stack = stack.fit(X_train, y_train)

# S_train = stack.transform(X_train)
# S_test = stack.transform(X_test)

# model = XGBClassifier(objective="binary:logistic",
#                     learning_rate=0.1,
#                     n_estimators=200,
#                     max_depth=2,
#                     subsample=0.7,
#                     random_state=123,
#                     colsample_bytree = 0.7,
#                     gamma = 0.1,
#                     min_child_weight = 6,
#                     reg_alpha = 0.01)

# pip_baseline = make_pipeline(RobustScaler(), model)
# scores_stk = cross_val_score(pip_baseline,
#                          X_train, y_train,
#                          scoring = "roc_auc", cv = 4)

# model = model.fit(S_train, y_train)

# y_pred = model.predict(S_test)


# # Testing the Classifier

test_df = pd.read_csv("tester_file.csv")

amount = int(input("Enter Requested Amount: "))
length_ = int(input("Enter Requested Loan Duration (in months, between 6 and 48): "))
rate = int(input("Requested Interest Rate: "))

# amount = amount.astype(int)
# length_ = length_.astype(int)
# rate = rate.astype(int)

interest  = amount * (rate/100) * (length_/12)

test_df.iloc[0, 5] = amount
test_df.iloc[0, 4] = length_
test_df.iloc[0, 6] = rate
test_df.iloc[0, 7] = interest

# test_df.to_csv("cust_file")

# Simple tester function

def cust_class(x):
    if x == 1:
        print("Congratulations! Your loan application has been APPROVED.")
    else:
        print("Sorry, Your loan application has been REJECTED.")

print("\n")
print("Result of Your Loan Application: ")
print("========================================================")
cust_class(logistic.predict(test_df))
print("\n")
print("\n")
print("\n")




