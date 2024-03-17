import pandas as pd
from sklearn.model_selection import cross_val_score
from dataset_loader import load_dataset,preprocessing
from all_metrics_append import metrics_append
from Boosting_models import MulticlassClassificationOvR, LogitBoost,MEBoost,AdaBoost,RUSBoost,GradientBoostingClassifier
from sklearn.preprocessing import (Normalizer, PowerTransformer, QuantileTransformer,
                                   PolynomialFeatures, KBinsDiscretizer, StandardScaler,
                                   MinMaxScaler, RobustScaler,LabelEncoder)
from sklearn.model_selection import StratifiedKFold
models = [LogitBoost,MEBoost,AdaBoost,RUSBoost,GradientBoostingClassifier]


dataset_names = ['Wine', 'Hayes_Roth', 'Contraceptive_Method_Choice',
                 'Pen-Based_Recognition_of_Handwritten_Digits', 'Vertebral_Column',
                 'Differentiated_Thyroid_Cancer_Recurrence', 'Dermatology',
                 'Balance_Scale', 'Glass_Identification', 'Heart_Disease',
                 'Car_Evaluation', 'Thyroid_Disease', 'Yeast',
                 'Page_Blocks_Classification', 'Statlog_Shuttle', 'Covertype']


def StandardScaler_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def MinMaxScaler_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def RobustScaler_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = RobustScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def Normalizer_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = Normalizer()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def PowerTransformer_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = PowerTransformer()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def QuantileTransformer_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = QuantileTransformer(output_distribution='uniform')
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test


def PolynomialFeatures_(X_train, X_test, y_train, y_test):
  le = LabelEncoder()
  y_train = le.fit_transform(y_train)
  y_test = le.transform(y_test)

  scaler = PolynomialFeatures(degree=2, include_bias=False)
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  return X_train, X_test, y_train, y_test

def KBinsDiscretizer_(X_train, X_test, y_train, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    scaler = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='uniform')
    X_train = scaler.fit_transform(X_train).toarray()  # Converting to a dense array
    X_test = scaler.transform(X_test).toarray()  # Converting to a dense array

    return X_train, X_test, y_train, y_test



preprocessors = [
    ('standard', StandardScaler_),
    ('minmax', MinMaxScaler_),
    ('robust', RobustScaler_),
    ('normalizer', Normalizer_),
    ('power', PowerTransformer_),
    ('quantile', QuantileTransformer_),
    ('polynomial', PolynomialFeatures_),
    ('kbins', KBinsDiscretizer_)
]
boosters_list = ["LogitBoost", "MEBoost", "AdaBoost", "RUSBoost", "GradientBoostingClassifier"]
metrics_all = {
    "Boosters":[],
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
best_combinations = {}
for dataset_name in dataset_names:
    print(f"Processing dataset: {dataset_name}")
    X, y = load_dataset(dataset_name)
    best_score = float('-inf')  # Start with the worst possible score
    best_combination = {'algorithm': None, 'preprocessor': None, 'score': None}

    for preprocessor_name, preprocessor in preprocessors:
        print(f"Applying preprocessor: {preprocessor_name}")

        for model in models:
            print(f"Evaluating model: {model.__name__}")
            mclf_boost_model = MulticlassClassificationOvR(model())
            metrics = metrics_append(X, y, mclf_boost_model, preprocessor, return_str=False)

            current_score = metrics['F1']
            if current_score > best_score:
                best_score = current_score
                best_combination['algorithm'] = model.__name__
                best_combination['preprocessor'] = preprocessor_name
                best_combination['score'] = current_score

            metrics_all["Boosters"].append(model.__name__)
            for metric, values in metrics.items():
                metrics_all[metric].append(values)
       

        pd.set_option('display.max_colwidth', None)
        df_metrics = pd.DataFrame(metrics_all)
        output_csv_path = f'tables/metrics_table_{dataset_name}.csv'
        df_metrics.to_csv(output_csv_path, index=False)

        best_combinations[dataset_name] = best_combination

df_best_combinations = pd.DataFrame.from_dict(best_combinations, orient='index')
output_path = 'best_algorithm_preprocessor_combinations.csv'
df_best_combinations.to_csv(output_path, index_label='Dataset')
print("Best combinations have been saved to:", output_path)
