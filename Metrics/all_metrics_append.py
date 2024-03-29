# -*- coding: utf-8 -*-
"""all_metrics_append.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MR5P26dJJDrA-1MO5a-nnqgk2uQQGJ1G
"""

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


def metrics_append (X, y, clf, preprocessing_local, return_str=True):
  metrics = {
    "Precision": [],
    "Recall": [],
    "F1": [],
    "G-mean": [],
    "MMCC": [],
    "Kappa": [],
    "Weighted Accuracy": [],
    "PR Score": [],
    "Balanced Accuracy": [],
    }

  # Setup for M-fold cross-validation
  skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

  for train_index, test_index in skf.split(X, y):
    X_train_fold = np.array(X)[train_index]
    y_train_fold = np.array(y)[train_index]
    X_test_fold = np.array(X)[test_index]
    y_test_fold = np.array(y)[test_index]

    X_train_fold, X_test_fold, y_train_fold, y_test_fold = preprocessing_local(X_train_fold, X_test_fold, y_train_fold, y_test_fold)

    # Initialize and fit your classifier
    clf.fit(X_train_fold, y_train_fold)

    # Predictions and scores
    y_pred_fold = clf.predict(X_test_fold)

    # Calculate metrics for this fold
    metrics["Precision"].append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=1))
    metrics["Recall"].append(recall_score(y_test_fold, y_pred_fold, average='weighted'))
    metrics["F1"].append(f1_score(y_test_fold, y_pred_fold, average='weighted'))
    metrics["G-mean"].append(hmean([precision_score(y_test_fold, y_pred_fold, average='weighted'), recall_score(y_test_fold, y_pred_fold, average='weighted')]))
    metrics["MMCC"].append(matthews_corrcoef(y_test_fold, y_pred_fold))
    metrics["Kappa"].append(cohen_kappa_score(y_test_fold, y_pred_fold))
    metrics["Weighted Accuracy"].append((accuracy_score(y_test_fold, y_pred_fold, normalize=False)) / len(y_test_fold))

    if len(np.unique(y)) > 2:
        y_test_fold_binarized = label_binarize(y_test_fold, classes=np.unique(y))

    else:
        y_test_fold_binarized = y_test_fold

    metrics["PR Score"].append(precision_score(y_test_fold, y_pred_fold, average='weighted', zero_division=1))

    metrics["Balanced Accuracy"].append(balanced_accuracy_score(y_test_fold, y_pred_fold))

  #for metric, values in metrics.items():
    #print(f"{metric}: {np.round(np.mean(values), 3)} ± {np.round(np.std(values), 3)}")
  if return_str:
      return {metric: f"{np.round(np.mean(values), 3)} ± {np.round(np.std(values), 3)}" for metric, values in metrics.items()}
  else:
      return {metric: np.mean(values) for metric, values in metrics.items()}
