from _functions import *
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
file_path = Path('Data/lending_data.csv')
df = pd.read_csv(file_path)
print(df.isnull().sum())
# Create features based on home ownership
# Transform homeowner column

df["homeowner"] = df["homeowner"].apply(changehomeowner)
df["loan_status"] = df["loan_status"].apply(loan_status)
x_cols = [i for i in df.columns if i not in ('loan_status')]
X = df[x_cols]
y = df['loan_status']
y.value_counts()
print('Check: count total variables in X and y datasets:')
def population_check(X,y):
    count = "{:,}".format(len(X))
    if len(X) == len(y):
        print(f'X and y variable counts match without error at {count} datasets')
    else:
        print('ERROR, recheck X and y variable counts..')
print('')
population_check(X,y)

# Create X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=1,
                                                    stratify=y)
#Scale the data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
print('Check x & y axis counts to ensure datasets are complete')
print('')
print(f'X_train has a shape of {X_train.shape}')
print('')
print(f'X_test has a shape of {X_test.shape}')
print('')
print(f'y_train has a shape of {y_train.shape}')
print('')
print(f'y_test has a shape of {y_test.shape}')

model_simple = LogisticRegression(solver='lbfgs', random_state=1)
model_simple.fit(X_train_scaled, y_train)
# Calculated the balanced accuracy score
y_pred = model_simple.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
# Display the confusion matrix
confusion_matrix(y_test, y_pred)
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))

ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train,y_train)
# View the count of target classes with Counter
Counter(y_resampled)
# Train the Logistic Regression model using the resampled data
model_naive = LogisticRegression(solver='lbfgs', random_state=1)
model_naive.fit(X_resampled,y_resampled)
# Calculated the balanced accuracy score
balanced_accuracy_score(y_test,y_pred)
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))
# Print the imbalanced classification report
print('')
print('Classification Report:')
print(classification_report_imbalanced(y_test,y_pred))

# Resample the training data with SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(
    X_train,y_train
)
# View the count of target classes with Counter
Counter(y_resampled)
# Train the Logistic Regression model using the resampled data
model_smote = LogisticRegression(solver='lbfgs', random_state=1)
model_smote.fit(X_resampled,y_resampled)
# Calculated the balanced accuracy score
y_pred = model_smote.predict(X_test)
balanced_accuracy_score(y_test,y_pred)
# Display the confusion matrix
confusion_matrix(y_test,y_pred)
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test,y_pred))
# Resample the training data with SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X,y)
# View the count of target classes with Counter
Counter(y_resampled)
# Train the Logistic Regression model using the resampled data
model_combo = LogisticRegression(solver='lbfgs', random_state=1)
model_combo.fit(X_resampled,y_resampled)
y_pred = model_combo.predict(X_test)
# Calculate the balanced accuracy score
balanced_accuracy_score(y_test,y_pred)
# Display the confusion matrix
confusion_matrix(y_test,y_pred)
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test,y_pred))

print('SMOTEENN, Cluster Centroids and SMOTE all had the highest balance accuracy score at 99.5%.')
print('SMOTE & SMOTEENN had the best recall scores at 100% for both low and high risk loans.')
print('SMOTE & SMOTEENN had the best geometric mean scores at 100% for both low and high risk loans.')