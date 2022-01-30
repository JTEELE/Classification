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


def population_check(X,y):
    count = "{:,}".format(len(X))
    if len(X) == len(y):
        print(f'X and y variable counts match without error at {count} datasets')
    else:
        print('ERROR, recheck X and y variable counts..')

def changehomeowner(homeowner):
    if homeowner == "own":
        return 0
    else:
        return 1
# Create features based on loan status  
def loan_status(loan_status):
    if loan_status == "low_risk":
        return 0
    else:
        return 1


def rename_months(issue_month):
    if issue_month == "Jan":
        return 1
    elif issue_month == "Feb":
        return 2
    elif issue_month == "Mar":
        return 3
    elif issue_month == "Apr":
        return 4
    elif issue_month == "May":
        return 5
    elif issue_month == "Jun":
        return 6
    elif issue_month == "Jul":
        return 7
    elif issue_month == "Aug":
        return 8
    elif issue_month == "Sep":
        return 9
    elif issue_month == "Oct":
        return 10
    elif issue_month == "Nov":
        return 11
    elif issue_month == "Dec":
        return 12
