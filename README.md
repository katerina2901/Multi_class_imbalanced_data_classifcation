# Boosting approaches with multi-label imbalanced data problem

This repository contains code for Final Project of ML Course in Skoltech.
To run this code and reproduce results you need install these packages and libraries, you should use Python 3.10.

## Project description 
Explore methods for improving the performance of boosting algorithms in multi-label classification tasks, especially handling unbalanced datasets. 

## Presentations
[Project presentation](https://docs.google.com/presentation/d/1oxwzHTHZcKOx6UTRPLHiuyml_b9V-wnnT7GAfO96uL8/edit#slide=id.p3) - official report on this project by our team.

## Repository structure
```
ML_Project
├── Datasets # From UCI Machine Learning Repository
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
│   ├── paper_experiment.txt
│   ├── 1st_experiment_base_learners.txt
│   ├── 2st_experiment_preprocessing.txt
│   └── 3st_experiment_resampling.txt
├── expirements # 
│   ├── paper_experiment. # repeat experiments from main paper with boosting methods
│   ├── 1st_experiment_base_learners.ipynb # experiments with another base estimators
│   ├── 2st_experiment_preprocessing.ipynb # experiments with preprocessing operators
│   └── 3st_experiment_resampling.ipynb # experiments with using data-level approach resampling methods
├── Boosting_models.py # implementation of boosting methods
├── all_metrics_append.py # function for evaluating metrics
├── all_metrics_tables.py # function for pritning metrics in tables
├── dataset_loader.py # function for loading dataset from UCI Machine Learning Repository
├── smote.py # ready implementation of smote from https://github.com/dialnd/imbalanced-algorithms/blob/master/smote.py
└── README.md
```
## Datasets 
All dataset from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) should be pre-processed with ```dataset_loader.py```. 

## Requirements
```!pip install ucimlrepo```
```!pip install catboost```


