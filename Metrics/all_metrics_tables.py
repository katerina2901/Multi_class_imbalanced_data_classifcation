# -*- coding: utf-8 -*-
"""all_metrics_tables.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MR5P26dJJDrA-1MO5a-nnqgk2uQQGJ1G
"""

#how to use (example):
#algorithms = [LogitBoost(), MEBoost(), AdaBoost(), RUSBoost(), GradientBoostingClassifier()]
#metrics_table(algorithms)



import pandas as pd
def metrics_table(algorithms):
  list_of_names = ['Wine', 'Hayes_Roth', 'Contraceptive_Method_Choice',
                 'Pen-Based_Recognition_of_Handwritten_Digits',
                 'Vertebral_Column', 'Differentiated_Thyroid_Cancer_Recurrence',
                 'Dermatology', 'Balance_Scale', 'Glass_Identification',
                 'Heart_Disease', 'Car_Evaluation', 'Thyroid_Disease', 'Yeast',
                 'Page_Blocks_Classification', 'Statlog_Shuttle', 'Covertype',
                 ]
  for data_name in list_of_names:
    X,y = load_dataset(data_name)
    metrics_all = {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "G-mean": [],
        "MMCC": [],
        "Kappa": [],
        "Weighted Accuracy": [],
        #"ROC AUC": [],
        "PR Score": [],
        "Balanced Accuracy": [],
        }
    for bin_clf in algorithms:
      mclf_boost_model = MulticlassClassificationOvR(bin_clf)
      metrics = metrics_append (X, y, mclf_boost_model, preprocessing, return_str=False)
      for metric, values in metrics.items():
        metrics_all[metric].append(values)


    pd.set_option('display.max_colwidth', None)
    df_metrics = pd.DataFrame(metrics_all)
    algorithm_names = [algorithm.__class__.__name__ for algorithm in algorithms]
    df_metrics.insert(0, 'Algorithm', algorithm_names)
    output_csv_path = f'tables/metrics_table_{data_name}.csv'
    df_metrics.to_csv(output_csv_path, index=False)
