from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score, accuracy_score)
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import hmean
import numpy as np
from tqdm.notebook import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score


def gmean_score(y_true, y_pred):
    return hmean([precision_score(y_true, y_pred, average='weighted'), recall_score(y_true, y_pred, average='weighted')])

def weighted_accuracy(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred, normalize=False)
    return acc / len(y_true)
# Setup for M-fold cross-validation
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
metrics = {
    "Precision": [],
    "Recall": [],
    "F1": [],
    "G-mean": [],
    "MMCC": [],
    "Kappa": [],
    "Weighted Accuracy": [],
    "ROC AUC": [],
    "PR Score": [],
    "Balanced Accuracy": [],
}

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]  # Direct indexing for y if it's a numpy array

    # Initialize and fit your classifier
    clf = AdaBoostClassifier(n_estimators=30, algorithm="SAMME", random_state=0)
    clf.fit(X_train_fold, y_train_fold)
    
    # Predictions and scores
    y_pred_fold = clf.predict(X_test_fold)
    y_pred_proba_fold = clf.predict_proba(X_test_fold)[:, 1]  # Assuming binary classification

    # Calculate metrics for this fold
    metrics["Precision"].append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=1))
    metrics["Recall"].append(recall_score(y_test_fold, y_pred_fold, average='weighted'))
    metrics["F1"].append(f1_score(y_test_fold, y_pred_fold, average='weighted'))
    metrics["G-mean"].append(gmean_score(y_test_fold, y_pred_fold))
    metrics["MMCC"].append(matthews_corrcoef(y_test_fold, y_pred_fold))
    metrics["Kappa"].append(cohen_kappa_score(y_test_fold, y_pred_fold))
    metrics["Weighted Accuracy"].append(weighted_accuracy(y_test_fold, y_pred_fold))
    
    if len(np.unique(y)) > 2:
        y_test_fold_binarized = label_binarize(y_test_fold, classes=np.unique(y))
        
    else:
        y_test_fold_binarized = y_test_fold

    metrics["ROC AUC"].append(roc_auc_score(y_test_fold_binarized, clf.predict_proba(X_test_fold), average='weighted', multi_class='ovr'))

    metrics["PR Score"].append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=1))

    metrics["Balanced Accuracy"].append(balanced_accuracy_score(y_test_fold, y_pred_fold))

def print_metrics(metrics):
    for metric, values in metrics.items():
        print(f"{metric}: {np.round(np.mean(values), 3)} ± {np.round(np.std(values), 3)}")

print_metrics(metrics)
