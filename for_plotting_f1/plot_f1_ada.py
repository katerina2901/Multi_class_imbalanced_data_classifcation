# -*- coding: utf-8 -*-
"""plot_f1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16CwJhT6rKd2MdE-k6Du_9GVveh9ybF0k
"""
import matplotlib.pyplot as plt
import pandas as pd
def plot_f1_score(file_path):
    df_base = pd.read_csv('logidboost_baseline.csv')
    df = pd.read_csv(file_path)
    plt.plot(range(len(df_base)), df_base["init_F1_mean"], color='grey', linestyle=':')
    plt.errorbar(x=range(len(df_base)), y=df_base["init_F1_mean"], 
                 yerr=df_base["init_F1_std"], fmt='o', 
                 color='grey', capsize=5,
                 # transform=trans1,
                 label='Baseline')
    #for your csv file
    #adjust color
    plt.plot(range(len(df_base)), df["F1_mean"], color='green', linestyle=':')
    plt.errorbar(x=range(len(df_base)), y=df["F1_mean"], yerr=df["F1_std"],
                 fmt='o', color='green',
                 capsize=5,
                 # transform=trans2,
                 label='With resampling')

    plt.xticks(range(len(df_base)), df_base['Dataset'], rotation=45, ha='right')

    plt.title('F1 Scores by Dataset with Standard Deviation for ADABoost')
    plt.ylabel('F1 Score')
    plt.xlabel('Datasets')
    plt.legend(fontsize='12')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('result.png')
    plt.show()

#how to use
#file_path = '/your_path'
#plot_f1_score(file_path)
