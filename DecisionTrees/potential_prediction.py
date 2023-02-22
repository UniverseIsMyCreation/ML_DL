import os

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline

import numpy as np


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        sum_empty_potential = 20 * 256

        for index in range(len(x)):
            potential_image = x[index]
            sum_rows = potential_image.sum(axis=1)
            sum_columns = potential_image.sum(axis=0)
            x_begin = (sum_rows == sum_empty_potential).argmin()
            x_end = 255 - (sum_rows == sum_empty_potential)[::-1].argmin()
            y_begin = (sum_columns == sum_empty_potential).argmin()
            y_end = 255 - (sum_columns == sum_empty_potential)[::-1].argmin()

            new_x = np.full((256, 256), 20)
            new_x_begin = (256 - (x_end - x_begin + 1)) // 2
            new_y_begin = (256 - (y_end - y_begin + 1)) // 2
            diff_x = x_end - x_begin + 1
            diff_y = y_end - y_begin + 1
            new_x[new_x_begin:new_x_begin+diff_x, new_y_begin:new_y_begin+diff_y] = potential_image[x_begin:x_end + 1, y_begin:y_end + 1]

            x[index] = new_x

        return x.reshape((x.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    better_model_forest = ExtraTreesRegressor(n_estimators=4500, max_depth=100, max_features=100)
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('decision_tree', better_model_forest)])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
