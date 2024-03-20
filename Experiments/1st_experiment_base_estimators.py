#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
import random
import copy
from tqdm.notebook import tqdm

from sklearn.model_selection import GridSearchCV, ParameterGrid
#estimators
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from smote import SMOTEBoost, SMOTE
from sklearn.ensemble import AdaBoostClassifier

from dataset_loader import load_dataset, preprocessing
from Boosting_models import MulticlassClassificationOvR, LogitBoost, MEBoost, AdaBoost, RUSBoost, GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.stats import hmean
from all_metrics_append import metrics_append


def find_best_base_estimator(X, y, boosting_model, base_estimators):
    best_params = {}

    for estimator_name, estimator in base_estimators.items():
        param_grid = {}

        if estimator_name in ['DecisionTreeClassifier', 'ExtraTreeClassifier', 'DecisionTreeRegressor']:
            param_grid = {
                'max_depth': [1, 2, 3, 4, 5],
                # 'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [2, 3],
                # 'max_features': ['auto', 'sqrt', 'log2']
            }
        elif estimator_name == 'LogisticRegression':
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                # 'penalty': ['l1', 'l2']
            }
        elif estimator_name == 'SVC':
            param_grid = {
                'C': [0.1, 1, 10],
                # 'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        else:
            continue

        param_grid = ParameterGrid(param_grid)
#         print(f"Parameter grid for {estimator_name}: {param_grid}")  # Debugging print
        f1_scores = []
        for param_set in param_grid:
            try: # some params are not possible together
                base_estimator = estimator(**param_set)
                if boosting_model == SMOTEBoost:
                    bin_clf = boosting_model(base_estimator=DecisionTreeClassifier())
                else:
                    bin_clf = boosting_model(base_estimator=base_estimator)
                model = MulticlassClassificationOvR(bin_clf)
                metrics = metrics_append(X, y, model, preprocessing, return_str=False)
                f1_score = metrics["F1"]
                f1_scores.append(f1_score)

            except Exception as e:  # Catch any exceptions :
#                 print(f"Error for {estimator_name} with params {param_set}: {e}")  # Debugging print
                continue

                f1_scores.append(-1) # in order to save indexing

        f1_scores = np.array(f1_scores)
        # print(f1_scores)
        best_index = np.argmax(f1_scores)
        best_params[estimator_name] = [f1_scores[best_index], param_grid[best_index]]
    best_score = 0
    best_estimator = None
    best_param = None
    for estimator_name in best_params:
        score = best_params[estimator_name][0]
        if score >= best_score:
            best_score = score
            best_estimator = estimator_name
            best_param = best_params[estimator_name][1]


    return best_estimator, best_param

def main():
    dataset_names = ['Wine', 'Hayes_Roth', 'Contraceptive_Method_Choice',
                    'Pen-Based_Recognition_of_Handwritten_Digits',
                    'Vertebral_Column', 'Differentiated_Thyroid_Cancer_Recurrence',
                    'Dermatology', 'Balance_Scale', 'Glass_Identification',
                    'Heart_Disease', 'Car_Evaluation', 'Thyroid_Disease', 'Yeast',
                     'Page_Blocks_Classification', 'Statlog_Shuttle', 'Covertype'
    ]


    base_estimators = {
        'DecisionTreeClassifier': DecisionTreeClassifier,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'ExtraTreeClassifier': ExtraTreeClassifier,
        'LogisticRegression': LogisticRegression,
        'SVC': SVC
    }

    boosting_models = {
        'AdaBoost': AdaBoost,
        'RUSBoost': RUSBoost, 
        'XGBClassifier': XGBClassifier,
        'CatBoostClassifier': CatBoostClassifier,
        'SMOTEBoost': SMOTEBoost,
        'LogitBoost': LogitBoost,
        'GradientBoostingClassifier':GradientBoostingClassifier,
        'MEBoost': MEBoost
    }


    for dataset_name in dataset_names:
        X, y = load_dataset(dataset_name)
        
        with open('results.txt', 'a') as f:
            f.write(f"DataSet: {dataset_name}\n")

        # Training boosting models on base models with the best parameters

        for boosting_name, boosting_model in boosting_models.items():
            try:

                best_estimator_name, best_param = find_best_base_estimator(X, y, boosting_model, base_estimators)
                base_estimator = base_estimators[best_estimator_name](**best_param)
                if boosting_model == SMOTEBoost:
                    bin_clf = boosting_model(base_estimator=base_estimator)
                else:
                    bin_clf = boosting_model(base_estimator=base_estimator)

                model = MulticlassClassificationOvR(bin_clf)
                metrics = metrics_append(X, y, model, preprocessing, return_str=True)
                
                print(boosting_name, best_estimator_name, best_param)
                print(metrics)
                with open('results.txt', 'a') as f:
                    f.write(f"{boosting_name} {best_estimator_name} {best_param}\n")
                with open('results.txt', 'a') as f:
                    f.write(f"Metrics: {metrics}\n")
            except:
                continue


if __name__ == "__main__":
    main()


# In[ ]:




