import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        X_numpy = X.to_numpy()
        self.features = list()
        for column in range(X.shape[1]):
            self.features.append(list(np.unique(X_numpy[:, column])))

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        one_hot_matrix = list()
        for row in X.values:
            features_of_one_elem = list()
            for index, elem in enumerate(row):
                idx_in_feature = self.features[index].index(elem)
                for i in range(len(self.features[index])):
                    if i == idx_in_feature:
                        features_of_one_elem.append(1)
                    else:
                        features_of_one_elem.append(0)
            one_hot_matrix.append(features_of_one_elem)

        return np.array(one_hot_matrix)

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        self.counters = list()
        for column in X.columns:
            dct = dict()
            features = np.unique(X[column])
            for feature in features:
                dct[feature] = list()
                amount = np.sum(X[column] == feature)
                weighted_amount = np.sum(Y[X[column] == feature])
                dct[feature].append(weighted_amount/amount)
                dct[feature].append(amount/X.shape[0])
            self.counters.append(dct)

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        counter_matrix = list()
        for row in X.values:
            features_of_one_elem = list()
            for index, elem in enumerate(row):
                success = self.counters[index][elem][0]
                counters = self.counters[index][elem][1]
                features_of_one_elem.append(success)
                features_of_one_elem.append(counters)
                features_of_one_elem.append((success+a)/(counters+b))
            counter_matrix.append(features_of_one_elem)

        return np.array(counter_matrix)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        self.training_splits = list()
        for group in group_k_fold(X.shape[0], n_splits=self.n_folds, seed=seed):
            self.training_splits.append(group)

        self.CounterEncoder = SimpleCounterEncoder()
        self.Y = Y

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        transformed_parts = list()
        for split in self.training_splits:
            self.CounterEncoder.fit(X.iloc[split[1]], self.Y.iloc[split[1]])
            transformed_part = self.CounterEncoder.transform(X.iloc[split[0]], a, b)
            transformed_parts.append([transformed_part, split[0]])

        if self.n_folds >= 2:
            counter_matrix = np.vstack((transformed_parts[0][0], transformed_parts[1][0]))
            index_matrix = np.hstack((transformed_parts[0][1].T, transformed_parts[1][1].T))
            for index in range(2, self.n_folds):
                counter_matrix = np.vstack((counter_matrix, transformed_parts[index][0]))
                index_matrix = np.hstack((index_matrix, transformed_parts[index][1]))
        else:
            counter_matrix = transformed_parts[0][0]
            index_matrix = transformed_parts[0][1]

        return counter_matrix[np.argsort(index_matrix)]

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    attributes = np.unique(x)
    weights = list()
    for elem in attributes:
        y_elem = y[elem == x]
        weights.append(np.sum(y_elem)/len(y_elem))
    return np.array(weights)
