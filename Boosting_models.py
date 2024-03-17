from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import numpy as np

import random

import copy

class MulticlassClassificationOvR:

    def __init__(self, bin_boosting_model):

        self.bin_boosting_model = bin_boosting_model

        self.models = []

    def fit(self, X, y):
        """
        Fits each model
        """
        for y_i in np.unique(y):
            # y_i - positive class for now
            # All other classes except y_i are negative

            # Choose x where y is positive class
            x_true = X[y == y_i]
            # Choose x where y is negative class
            x_false = X[y != y_i]
            # Concatanate
            x_true_false = np.vstack((x_true, x_false))

            # Set y to 1 where it is positive class
            y_true = np.ones(x_true.shape[0])
            # Set y to 0 where it is negative class
            y_false = np.zeros(x_false.shape[0])
            # Concatanate
            y_true_false = np.hstack((y_true, y_false))

            # Fit model and append to models list
            model = copy.copy(self.bin_boosting_model)
            # model = self.bin_boosting_model
            model.fit(x_true_false, y_true_false)
            self.models.append([y_i, model])

    def predict(self, X):
        y_pred = [[label, model.predict(X)] for label, model in self.models]

        output = []

        for i in range(X.shape[0]):
            max_label = None
            max_prob = -10**5
            for j in range(len(y_pred)):
                prob = y_pred[j][1][i]
                if prob > max_prob:
                    max_label = y_pred[j][0]
                    max_prob = prob
            output.append(max_label)

        return output

##############################################################################

class LogitBoost:

    def __init__(self,  base_estimator=None, learning_rate=0.1, n_estimators=50,
                 max_depth=3, random_state=0, sampler = None, sampler_type="Global"):

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        if base_estimator:
            self.base_estimator = base_estimator
        else:
            self.base_estimator = DecisionTreeRegressor(max_depth=self.max_depth, criterion='friedman_mse', random_state=self.random_state)

        if sampler:
          # print("Sampling is used")
          self.sampler_flag = True
          self.sampling = sampler
          self.sampler_type = sampler_type

        else:
          self.sampler_flag = False
          self.samler_type = "None"        

    def _softmax(self, predictions):
        exp = np.exp(predictions)

        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, Y):

        if self.sampler_flag == True and self.sampler_type == "Global":
          # print("Oversampling")
          try:
            X, Y = self.sampling.fit_resample(X, Y)
          except:
            sampled = self.sampling(X, Y)
            X = np.array([s[1] for s in sampled])
            Y = np.array([s[2] for s in sampled])

        predictions = np.zeros((Y.shape[0], 2))

        self.trees1 = []

        for _ in range(self.n_estimators):
            probabilities = self._softmax(predictions)

            numerator = (Y.T - probabilities.T[0])
            denominator = probabilities.T[0] * (1 - probabilities.T[0])
            residuals = 1 / 2 * numerator / denominator
            weights = denominator

            tree = copy.copy(self.base_estimator)

            tree.fit(X, residuals, sample_weight=weights)
            self.trees1.append(tree)

            y_pred = tree.predict(X)

            predictions.T[0] += self.learning_rate * y_pred
            predictions.T[1] += self.learning_rate * (1-y_pred)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], 2))

        for i in range(self.n_estimators):
            y_pred_pos = self.trees1[i].predict(X)
            y_pred_neg = 1 - y_pred_pos
            predictions.T[0] += self.learning_rate * y_pred_neg
            predictions.T[1] += self.learning_rate * y_pred_pos

        return np.argmax(predictions, axis=1)

############################################        

class MEBoost:
    def __init__(self, base_estimator=None, n_estimators=50,
     learning_rate=1, sampler = None, sampler_type="Global"):

        if base_estimator:
            self.base_estimator = base_estimator
        else:
            self.base_estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)

        if sampler:
          # print("Sampling is used")
          self.sampler_flag = True
          self.sampling = sampler
          self.sampler_type = sampler_type

        else:
          self.sampler_flag = False
          self.samler_type = "None"

        self.n_estimators = n_estimators

        self.estimators = None
        self.estimator_weights = None

    def fit(self, X, Y):

        if self.sampler_flag == True and self.sampler_type == "Global":
          # print("Oversampling")
          try:
            X, Y = self.sampling.fit_resample(X, Y)
          except:
            sampled = self.sampling(X, Y)
            X = np.array([s[1] for s in sampled])
            Y = np.array([s[2] for s in sampled])

        X, X_test_loc, Y, Y_test_loc = train_test_split(X, Y, test_size=0.3,
                                                    random_state=42, stratify=Y)

        #  resetting lists
        self.estimators = []
        self.estimator_weights = []

        models = []
        alphas = []

        best_alpha = []
        best_tree = []

        top_score = 0
        nonImproving = 0

        #  0) initialise equal weights
        sample_weights = np.full(len(X), 1/len(X))
        m = 0

        while True:
            m += 1

            while True:

              if self.sampler_flag == True and self.sampler_type == "Boost_step":
                # print("Undersampling")
                sampled = self.sampling(X, Y)
                sampled_weight = [sample_weights[s[0]] for s in sampled]
                sampled_X      = [s[1] for s in sampled]
                sampled_Y       = [s[2] for s in sampled]
                sampled_indeces = [s[0] for s in sampled]

              else:
                sampled_X = X
                sampled_Y = Y
                sampled_indeces = range(0, X.shape[0])
                sampled_weight = sample_weights

              sampled_weight = np.array(sampled_weight)
              sampled_X = np.array(sampled_X)
              sampled_Y = np.array(sampled_Y)

              #  1) fit weak learner
              if m % 2 == 0:
                estimator = copy.copy(self.base_estimator)
              else:
                estimator = ExtraTreeClassifier(max_depth=2, min_samples_split=4)

              # estimator = copy.copy(self.base_estimator)
              estimator.fit(sampled_X, sampled_Y, sampled_weight)

              #  2) calculate total error
              prediction = estimator.predict(sampled_X)
              total_error = np.where(prediction != sampled_Y, sampled_weight, 0).sum()

              if total_error <= 0:
                total_error = 1e-9

              if total_error < 0.5:
                break

            alpha = 0.5 * np.log((1 - total_error)/(total_error + 1e-10))

            #  4) update weights
            sampled_weight = np.where(prediction != sampled_Y, sampled_weight * np.exp(alpha), sampled_weight * np.exp(-1 * alpha))

            #  5) renormalize weights
            sampled_weight = sampled_weight / sampled_weight.sum()

            sample_weights[sampled_indeces] = sampled_weight

            FX = np.zeros(X_test_loc.shape[0])
            FX += estimator.predict(X_test_loc)*alpha
            for alpha_prev, tree in zip(self.estimator_weights, self.estimators):
                FX += alpha_prev * tree.predict(X_test_loc)
            FX = np.sign(FX)

            score = roc_auc_score(Y_test_loc, FX)
            if top_score <= score:
                top_score = score
                self.estimators.append(estimator)
                self.estimator_weights.append(alpha)

            if score <= top_score:
                nonImproving += 1
                if nonImproving == 100:

                  break

    def predict(self, X, verbose=False):
        """
        * every estimator makes his predictions in the shape (len(X)) -> [a, b, ..., len(X)]
        * stack prediction of estimators to have them row wise(each row corresponds to a sample) -> [[a1, a2], [b1, b2], ..., len(X)]
        * at each row apply the weighted majority vote previously discussed
        """
        predictions = np.stack([estimator.predict(X) for estimator in self.estimators], axis=1)
        weighted_majority_vote = lambda x: np.unique(x)[np.argmax([np.where(x==categ, self.estimator_weights, 0).sum() for categ in np.unique(x)])]
        return np.apply_along_axis(weighted_majority_vote, axis=1, arr=predictions)


############################################# AdaBoost ##############################

def I(flag):
    return 1 if flag else 0

def sign(x):
    return abs(x)/x if x!=0 else 1

class AdaBoost:

    def __init__(self, base_estimator=None, n_estimators=50, sampler = None, sampler_type="Global"):
        self.n_estimators = n_estimators

        if base_estimator:
            # print("Estimator is implemented")
            self.base_estimator = base_estimator
        else:
            self.base_estimator = DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2)

        if sampler:
          # print("Sampling is used")
          self.sampler_flag = True
          self.sampling = sampler
          self.sampler_type = sampler_type

        else:
          self.sampler_flag = False
          self.samler_type = "None"

        self.models = [None]*n_estimators

    def fit(self,X,Y):

        if self.sampler_flag == True and self.sampler_type == "Global":
          # print("Oversampling")
          try:
            X, Y = self.sampling.fit_resample(X, Y)
          except:
            sampled = self.sampling(X, Y)
            X = np.array([s[1] for s in sampled])
            Y = np.array([s[2] for s in sampled])

        self.models = [None]*self.n_estimators

        Y = np.where(Y==0,-1,1)

        X = np.float64(X)
        # N = len(Y)
        # w = np.array([1/N for i in range(N)])
        sample_weights = np.full(len(X), 1/len(X))

        for m in range(self.n_estimators):
            if self.sampler_flag == True and self.sampler_type == "Boost_step":
              # print("Undersampling")
              sampled = self.sampling(X, Y)
              sampled_weight = [sample_weights[s[0]] for s in sampled]
              sampled_X      = [s[1] for s in sampled]
              sampled_Y       = [s[2] for s in sampled]
              sampled_indeces = [s[0] for s in sampled]

            else:
              sampled_X = X
              sampled_Y = Y
              sampled_indeces = range(0, X.shape[0])
              sampled_weight = sample_weights

            sampled_weight = np.array(sampled_weight)
            sampled_X = np.array(sampled_X)
            sampled_Y = np.array(sampled_Y)

            model = copy.copy(self.base_estimator)

            model.fit(X, Y, sample_weight=sampled_weight)

            y_pred = model.predict(X)

            total_error = np.where(y_pred != Y, sampled_weight, 0).sum()
            # Gm = copy.copy(self.base_estimator).fit(X,y,sample_weight=w).predict

            # errM = sum([w[i]*I(y[i]!=Gm(X[i].reshape(1,-1))) \
            #             for i in range(N)])/sum(w)

            # AlphaM = np.log((1-errM)/errM)

            # 0.5 * 
            alpha = 0.5 * np.log((1 - total_error)/(total_error + 1e-10))

            # w = [w[i]*np.exp(AlphaM*I(y[i]!=Gm(X[i].reshape(1,-1))))\
            #          for i in range(N)]

            sampled_weight = np.where(y_pred != Y, sampled_weight * np.exp(alpha), sampled_weight * np.exp(-1 * alpha))

            sampled_weight = sampled_weight / sampled_weight.sum()
            sample_weights[sampled_indeces] = sampled_weight

            self.models[m] = (alpha, model)

    def predict(self,X):

        y = 0
        for m in range(self.n_estimators):
            alpha, model = self.models[m]
            y += alpha*model.predict(X)

        signA = np.vectorize(sign)
        y = np.where(signA(y)==-1,0,1)
  
        return y

##########################################################################
from collections import Counter
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class RUSBoost:

    def __init__(self, base_estimator=None, n_estimators=500, learning_rate=1e-2, seed=42):
        if base_estimator:
          self.base_estimator = base_estimator
        else:
          self.base_estimator = DecisionTreeClassifier(max_depth=1)  #SVC() #LogisticRegression()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []  # List to store models
        set_seed(seed)

    def fit(self, X, y):
        self.models = []
        self.estimator_weights = np.zeros(self.n_estimators)  # Initialize estimator weights
        self.estimator_errors = np.zeros(self.n_estimators)  # Initialize estimator errors
        sample_weights = np.full(X.shape[0], 1 / X.shape[0])  # create an array with shape X.shape[0] and value 1 / X.shape[0]

        for i in range(self.n_estimators):
            # Undersampling
            X_resampled, y_resampled = self.undersampling(X, y, sample_weights)

            # Train a base estimator on the resampled data
            model = copy.copy(self.base_estimator)
            model.fit(X_resampled, y_resampled)
            self.models.append(model)

            # Update sample weights
            y_pred = model.predict(X)
            errors = np.abs(y_pred - y)
            self.estimator_errors[i] = np.sum(sample_weights * errors) / np.sum(sample_weights)  # Calculate error
            self.estimator_weights[i] = self.learning_rate * np.log((1 - self.estimator_errors[i]) / self.estimator_errors[i])  # Update weight

            # Update sample weights
            sample_weights *= np.exp(self.estimator_weights[i] * errors)
            sample_weights /= np.sum(sample_weights)  # Normalize weights

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model, weight in zip(self.models, self.estimator_weights):
            predictions += weight * model.predict(X)
        return np.where(predictions >= 0.5 * sum(self.estimator_weights), 1, 0)

    def undersampling(self, X, y, sample_weights):

        # count the number of examples for each class
        class_counts = Counter(y)

        # find the minority class
        minority_class = min(class_counts, key=class_counts.get)
        minority_count = class_counts[minority_class]
        keep_list = [(i, X[i], label) for i, label in enumerate(y) if label == minority_class]

        # find other classes as major
        delete_list = [(i, X[i], label) for i, label in enumerate(y) if label != minority_class]
        delete_indices = random.sample(range(len(delete_list)), k=max(0, len(delete_list) - (len(delete_list)-minority_count)) )
        keep_list_not_minor = [delete_list[i] for i in delete_indices]

        all_list = keep_list + keep_list_not_minor
        all_list.sort(key=lambda x: x[2])

        X_resampled = np.array([item[1] for item in all_list])
        y_resampled = np.array([item[2] for item in all_list])

        return X_resampled, y_resampled

##########################################################################

class GradientBoostingClassifier:
    def __init__(self, base_estimator=None, n_estimators=100,
     learning_rate=1.0, max_depth=3,
      min_samples_split=2, loss='deviance',
       seed=None, sampler = None):

        self.base_estimator = base_estimator if base_estimator else DecisionTreeRegressor(max_depth=1, max_leaf_nodes=2, random_state=42)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.seed = seed
        self.models = []
        if seed is not None:
            np.random.seed(seed)

        if sampler:
          # print("Sampling is used")
          self.sampler_flag = True
          self.sampling = sampler

        else:
          self.sampler_flag = False
        
    def _loss_derivative(self, y, pred):
        """Compute the derivative of the loss function."""
        # Using logistic loss derivative for binary classification
        return y - 1 / (1 + np.exp(-pred))

    def fit(self, X, y):
        # Convert y to {0, 1}
        if self.sampler_flag == True:
          # print("Oversampling")
          try:
            X, y = self.sampling.fit_resample(X, y)
          except:
            sampled = self.sampling(X, y)
            X = np.array([s[1] for s in sampled])
            y = np.array([s[2] for s in sampled])

        self.models = []
        y = (y == 1).astype(int)

        # Initialize with a dummy model that predicts the log odds ratio
        initial_pred = np.log(y.mean() / (1 - y.mean()))
        f_m = np.full(shape=y.shape, fill_value=initial_pred)
        self.initial_model = initial_pred

        for i in range(self.n_estimators):
            # Compute pseudo-residuals as negative gradient of the loss function
            residuals = self._loss_derivative(y, f_m)

            # Fit a base model to the residuals
            # model = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            model = copy.copy(self.base_estimator)
            model.fit(X, residuals)
            self.models.append(model)

            # Update the predictions
            f_m += self.learning_rate * model.predict(X)

    def predict_proba(self, X):
        # Compute the sum of predictions from the initial model and all subsequent models
        f_m = np.full(shape=(X.shape[0],), fill_value=self.initial_model)
        for model in self.models:
            f_m += self.learning_rate * model.predict(X)

        # Convert the log odds ratio to probabilities
        proba = 1 / (1 + np.exp(-f_m))
        return np.vstack((1-proba, proba)).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

##########################################################################

