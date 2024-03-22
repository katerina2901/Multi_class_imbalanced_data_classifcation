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
│   ├── best_algorithm_preprocessor_summary.csv # contain 3 st_experiment best results for all datasets
│   └── final_result.csv # contain a comparison of the best results from all experiments for all datasets
├── README.md
└── dataset_loader.py # function for loading dataset from UCI Machine Learning Repository
```
## Total result
The best results were obtained by selection the best estimators, including the searching of hyperparameters for the estimators hence Hypothesis 1 is confirmed.

The conducted experiments have shown that the hypothesis about algorithms that are robust to imbalanced data should yield better results is inaccurate.
Total result for all datasets:
```
Dataset,MeanIR,Top Algorithm,F1 best performance,F1 baseline
Pen-Based,0.961,AdaBoost,0.995 ± 0.007,0.989 ± 0.001
Hayes,0.856,AdaBoost,0.908 ± 0.037,0.891 ± 0.024
Wine,0.836,AdaBoost,0.989 ± 0.015,0.983 ± 0.024
Contraceptive,0.781,GradientBoost,0.681 ± 0.134,0.461 ± 0.061
Balance_Scale,0.723,AdaBoost,0.956 ± 0.007,0.857 ± 0.025
Differentiated,0.696,GradientBoost,0.987 ± 0.009,0.979 ± 0.015
Vertebral,0.689,AdaBoost,0.904 ± 0.055,0.862 ± 0.016
Dermatology,0.538,AdaBoost,0.987 ± 0.016,0.98 ± 0.017
Thyroid,0.478,AdaBoost,0.981 ± 0.018,0.962 ± 0.014
Glass,0.469,AdaBoost,0.841 ± 0.079,0.744 ± 0.104
Heart,0.371,AdaBoost,0.824 ± 0.248,0.561 ± 0.11
Car,0.357,AdaBoost,0.974 ± 0.008,0.974 ± 0.008
Yeast,0.321,AdaBoost,0.836 ± 0.202,0.582 ± 0.039
Statlog,0.182,GradientBoost,0.998 ± 0.0,0.998 ± 0.0
Covertype,0.293,GradientBoost,0.805 ± 0.005,0.87 ± 0.0
```


## Requirements
```!pip install ucimlrepo```

```!pip install catboost```


