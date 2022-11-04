import numpy as np
from collections import defaultdict


def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    onefold_length = num_objects // num_folds
    kfold_split_list = []
    for index in range(num_folds):
        if index == num_folds-1:
            if num_objects % num_folds:
                start, end = index*onefold_length, (index+1)*onefold_length + num_objects % num_folds
            else:
                start, end = index*onefold_length, (index+1)*onefold_length
        else:
            start, end = index*onefold_length, (index+1)*onefold_length
        one_fold = []
        one_fold.append(np.arange(start, end))
        train_indexes = np.concatenate((np.arange(start), np.arange(end, num_objects)))
        one_fold.append(train_indexes)
        one_fold[0], one_fold[1] = one_fold[1], one_fold[0]
        one_fold = tuple(one_fold)
        kfold_split_list.append(one_fold)
    return kfold_split_list


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations)

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    results = {}
    for n_neighbor in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weight in parameters['weights']:
                kNN = knn_class(n_neighbors=n_neighbor, weights=weight, metric=metric)
                for normalizer in parameters['normalizers']:

                    prediction_score_sum = 0

                    for fold in folds:

                        if normalizer[0] is None:
                            normalized_X = X
                        else:
                            scaler = normalizer[0]
                            scaler.fit(X[fold[0]])
                            normalized_X = scaler.transform(X)

                        kNN.fit(normalized_X[fold[0]], y[fold[0]])
                        y_prediction = kNN.predict(normalized_X[fold[1]])
                        prediction_score_sum += score_function(y[fold[1]], y_prediction)

                    prediction_score_sum /= len(folds)
                    results[(normalizer[1], n_neighbor, metric, weight)] = prediction_score_sum
    return results
