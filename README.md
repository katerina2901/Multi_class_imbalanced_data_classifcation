# Boosting approaches with multi-label imbalanced data problem

This repository contains code for Final Project of ML Course in Skoltech.
To run this code and reproduce results you need install these packages and libraries, you should use Python 3.10.

## Project description 
Explore methods for improving the performance of boosting algorithms in multi-label classification tasks, especially handling unbalanced datasets. 

## Presentations
[Project presentation](https://docs.google.com/presentation/d/1oxwzHTHZcKOx6UTRPLHiuyml_b9V-wnnT7GAfO96uL8/edit#slide=id.p1) - official report on this project by our team.

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
│   ├── Metrics.py # function for evaluating metrics
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
│   ├── baseline_experiment.csv
│   ├── 1st_experiment_base_estimators.csv
│   ├── 2nd_experiment_preprocessing.csv
│   └── 3st_experiment_resampling.txt
├── dataset_loader.py # function for loading dataset from UCI Machine Learning Repository
└── README.md
```
## Datasets 
All dataset from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) should be pre-processed with ```dataset_loader.py```. 

## Requirements
```!pip install ucimlrepo```

```!pip install catboost```


