# Machine Learning - Classification

# See writeup file (pdf) for support of model results
This assignment utilized binary classification to determine whether loans were high or low risk. 

## Resample

Three resample methods were used for the resample portion of this assignment to determine *high-risk loans*:
Standard Scaler for oversampling: Balance accuracy: 94%; Precision (0,1): 100%/87%; Recall(0,1): 100%/89%; Specificity(0,1): 89%/100%. 

Over Sampler for oversampling: Balance accuracy: 94%; Precision (0,1): 100%/87%; Recall(0,1): 100%/89%; Specificity(0,1): 89%/100%.

SMOTE for both oversampling and undersampling: Balance accuracy: 99.5%; Precision (0,1): 100%/87%; Recall(0,1): 100%/100%; Specificity(0,1): 100%/100%.

Cluster Centroids for undersampling: Balance accuracy: 99.5%; Precision (0,1): 100%/87%; Recall(0,1): 100%/98%; Specificity(0,1): 98%/100%.

SMOTEENN for both oversampling and undersampling: Balance accuracy: 99.5%; Precision (0,1): 100%/87%; Recall(0,1): 100%/100%; Specificity(0,1): 100%/100%.

## Ensemble
Two ensemble classifiers were used to determine whether loans were high or low risk.

BalancedRandomForestClassifier: Balance accuracy: 88%; Precision (0,1): 100%/3%; Recall(0,1): 88%/72%; Specificity(0,1): 72%/88%.

EasyEnsembleClassifier: Balance accuracy: 85%; Precision (0,1): 100%/3%; Recall(0,1): 85%/83%; Specificity(0,1): 83%/85%.

## Analysis
Resample: All resample methods are effective for the linear regression model, although SMOTEENN outperforms the other resample methods as it produced the linear model with the highest accuracy rate.

Ensemble: Both BalancedRandomForestClassifier & EasyEnsembleClassifier had favoriable results, however the BalancedRandomForestClassifier would be preferred as the classification report produced more favorable results.


