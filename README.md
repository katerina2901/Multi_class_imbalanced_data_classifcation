# Boosting approaches with multi-label imbalanced data problem

This repository contains code for Final Project of ML Course in Skoltech.
To run this code and reproduce results you need install these packages and libraries, you should use Python 3.10.

## Project description 
Learning on skewed datasets becomes very important since many real-world classification tasks are usually unbalanced. Multiclass unbalanced learning is even more challenging than binary scenarios and remains an open problem. Busting algorithms that improve model performance by combining underlying learners face difficulties in obtaining good predictions on large unbalanced multiclass datasets. 
The goal of this project is to conduct experiments aimed at improving the performance of Boosting algorithms in multi-class classification tasks. The main contribution of this report is the experiments:

- Base-Line Implementation

As a baseline all considered boosting algorithms were used with default hypo-parameters. To estimate the performance of each algorithm  such metrics were used as: Precision, Recall, F1, 	G-mean (Geometric Mean of Sensitivity and Specificity), MMCC (Multi- class Matthews Correlation Coefficient), Kappa, Weighted Accuracy, PR Score, Balanced Accuracy.
- Studying the influence of base learners within the bousting procedure

In this [expirement](https://github.com/katerina2901/Multi_class_imbalanced_data_classifcation/blob/main/Experiments/1st_experiment_base_learners.py) we used base estimators such as : Desicion Tree Regressor, Desicion Tree Classifier, Extra Tree Classifier, Support Vector Classifier and Logistic Regression. For each algorithm, the base estimator with the best hyper parameters was selected using ParameterGrid. 
- Studying the effect of preprocessing operators on the performance of boosting algorithms.

[Preprocessing]([https://github.com/katerina2901/Multi_class_imbalanced_data_classifcation/blob/main/Experiments/1st_experiment_base_learners.py)](https://github.com/katerina2901/Multi_class_imbalanced_data_classifcation/blob/main/Experiments/2st_experiment_preprocessors.py) employs several boosting models, including `LogitBoost`, `MEBoost`, `AdaBoost`, `RUSBoost`, and `GradientBoostingClassifier`, to iteratively improve predictions by focusing on previous errors. 
- Implementation of combined ensemble methods using data-level sampling techniques
In the [expirement]([https://github.com/katerina2901/Multi_class_imbalanced_data_classifcation/blob/main/Experiments/Experiment_3_resampling.ipynb])  we tried to improve the performance of boosting algorithms by balancing classes using various resampling techniques like Random oversampling, Borderline SMOTE, Random undersampling, NearMiss-1, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, ALLKNN, OneSidedSelection and NeighborhoodCleaningRule.

  Our hypotheses:
- There is a significant difference in performance between different boosting algorithms when using different estimators.
- Algorithms that are more robust to imbalanced data, like RusBoost and SmoteBoost, should yield better results.

## Datasets 
All dataset from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) should be pre-processed with ```dataset_loader.py```. 

## Presentations
[Project presentation](https://github.com/katerina2901/Multi_class_imbalanced_data_classifcation/blob/main/report/presentation.pdf) - official report on this project by our team.

## Repository structure
```
ML_Project
├── Boosting models # Contain scripts with boosting model mplementation
│   ├── Boosting_models.py # Implemented boosting models
│   └── smote.py # ready implementation of smote from https://github.com/dialnd/imbalanced-algorithms/blob/master/smote.py
├── Experiments 
│   ├── baseline.py # repeat experiments from main paper with boosting methods
│   ├── 1st_experiment_base_estimators.py # experiments with different base estimators
│   ├── 2st_experiment_preprocessors.py # experiments with different preprocessing operators
│   └── 3st_experiment_resampling.py # experiments with using data-level approach resampling methods
├── Metrics 
│   ├── all_metrics_append.py # function for evaluating all metrics
│   └── all_metrics_tables.py # function for pritning metrics in tables
├── Datasets # Dataset from UCI Machine Learning Repository that can't be imported in python
│   ├── hayes_roth.data
│   ├── new-thyroid.data
│   ├── page_blocks.data
│   ├── shuttle.trn
│   ├── shuttle.tst
│   └── vertebral.dat
├── report # report deliverables
│   ├── presentation.pdf
│   └── report.pdf
├── results # results of expirements
│   ├── 1st_experiment_base_estimators.csv # contain 1st_experiment best results for all datasets
│   ├── 2nd_experiment_preprocessors.csv # contain 2st_experiment all results for all datasets
│   ├──  3rd_experiment-preprocessors_all.zip # contain 3rd_experiment all results for all datasets
│   ├── baseline_best_algorithms.csv # contain baseline best results for all datasets
│   ├── baseline_experiment.csv # contain baseline results for models without hyperparameter tuning
│   └── best_algorithm_preprocessor_summary.csv # contain 3 st_experiment best results for all datasets
├── README.md
└── dataset_loader.py # function for loading dataset from UCI Machine Learning Repository
```

## Requirements
```!pip install ucimlrepo```

```!pip install catboost```


