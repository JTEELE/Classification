from _functions import *

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Load the data
file_path = Path('Data/LoanStats_2019Q1.csv')
df = pd.read_csv(file_path)

# Preview the data
print(df.isnull().sum())
df.head()


#Clean data for model; Convert issue date to numerical format
df['issue_month'] = df['issue_d'].str.split('-').str[0]
    
df["issue_month"] = df["issue_month"].apply(rename_months)
#Alternative Method, use get_dummies
#     df = pd.get_dummies(df, columns=['home_ownership','verification_status','initial_list_status','application_type'])

le = LabelEncoder()
# Encoding home_ownership column
le.fit(df["home_ownership"])
df["home_ownership"] = le.transform(df["home_ownership"])
# Encoding initial_list_status column
le.fit(df["verification_status"])
df["verification_status"] = le.transform(df["verification_status"])
# Encoding initial_list_status column
le.fit(df["initial_list_status"])
df["initial_list_status"] = le.transform(df["initial_list_status"])
# Encoding application_type column
le.fit(df["application_type"])
df["application_type"] = le.transform(df["application_type"])
#Format target variable
def loan_status(loan_status):
    if loan_status == "low_risk":
        return 0
    else:
        return 1
df["loan_status"] = df["loan_status"].apply(loan_status)
#Drop columns in which values are consistent for entire dataset. There is no value in using these features.
df.drop(columns=['pymnt_plan','hardship_flag','debt_settlement_flag','issue_d','next_pymnt_d'], inplace=True)
# Create our features
x_cols = [i for i in df.columns if i not in ('loan_status')]
X = df[x_cols]
# Create our target
y = df['loan_status']
X.describe()


population_check(X,y)

# Split the X and y into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=1,
                                                    stratify=y)


# Create the StandardScaler instance
scaler = StandardScaler()
# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
X_scaler = scaler.fit(X_train)
# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
# Resample the training data with the BalancedRandomForestClassifier
clf_model = BalancedRandomForestClassifier(random_state=1)
clf_model = clf_model.fit(X_train_scaled,y_train)
y_predict = clf_model.predict(X_test_scaled)
results = pd.DataFrame({'Predictions': y_predict, 'Actual': y_test}).reset_index(drop=True)

# Calculated the balanced accuracy score
bas = balanced_accuracy_score(y_test,y_predict)


# Display the confusion matrix
# Calculating the confusion matrix
cm = confusion_matrix(y_test, y_predict)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)
# Calculating the accuracy score
acc_score = accuracy_score(y_test, y_predict)
# Displaying results
print("Confusion Matrix")
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, y_predict))

# Print the imbalanced classification report
print(classification_report_imbalanced(y_test,y_predict))



# List the features sorted in descending order by feature importance
feature_importance = clf_model.feature_importances_
feature_importance_sorted = sorted(zip(clf_model.feature_importances_, X.columns), reverse=True)
feature_importance_sorted[:10]



# Train the Classifier
eec_model = EasyEnsembleClassifier(random_state=1)
eec_model = eec_model.fit(X_train_scaled,y_train)
y_predict = eec_model.predict(X_test_scaled)
results = pd.DataFrame({'Predictions': y_predict, 'Actual': y_test}).reset_index(drop=True)
results.head(3)

# Calculated the balanced accuracy score
bas = balanced_accuracy_score(y_test,y_predict)

# Display the confusion matrix
# Calculating the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_predict)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]
)

# Calculating the accuracy score
acc_score = accuracy_score(y_test, y_predict)

# Displaying results
print("Confusion Matrix")
print(f"Accuracy Score : {acc_score}")
print("Classification Report")
print(classification_report(y_test, y_predict))


# Print the imbalanced classification report
print(classification_report_imbalanced(y_test,y_predict))


print('Results:')
print('SMOTEENN, Cluster Centroids and SMOTE all had the highest balance accuracy score at 99.5%.')
print('SMOTE & SMOTEENN had the best recall scores at 100% for both low and high risk loans.')
print('SMOTE & SMOTEENN had the best geometric mean scores at 100% for both low and high risk loans.')