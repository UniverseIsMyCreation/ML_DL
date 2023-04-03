import pandas as pd
from numpy import ndarray
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor


def preprocessing_data(train, test):

    str_cols = ['genres', 'directors', 'filming_locations', 'keywords']
    bin_str_cols = ['actor_2_gender', 'actor_1_gender', 'actor_0_gender']

    for str_col in str_cols:
        text_train = train[str_col]
        text_test = test[str_col]
        text_train = text_train.apply(lambda x: (' '.join(str(word).lower() for word in x)).split(' '))
        text_test = text_test.apply(lambda x: (' '.join(str(word).lower() for word in x)).split(' '))
        vocabulary = set()
        for row in text_train:
            for word in row:
                if len(word) > 0:
                    vocabulary.add(word)
        vocabulary = list(vocabulary)
        text_train = text_train.apply(lambda x: ' '.join(word for word in x))
        text_test = text_test.apply(lambda x: ' '.join(word for word in x))
        pipe = Pipeline([
            ('count', CountVectorizer(vocabulary=vocabulary)),
            ('tfid', TfidfTransformer())
            ])
        pipe.fit(text_train)
        new_text_train = pipe.transform(text_train).todense()
        new_text_test = pipe.transform(text_test).todense()
        for idx, row in enumerate(new_text_train):
            train.at[idx, str_col] = float(np.sum(row))
        for idx, row in enumerate(new_text_test):
            test.at[idx, str_col] = float(np.sum(row))

    for str_col in bin_str_cols:
        train[str_col] = train[str_col].astype('category')
        test[str_col] = test[str_col].astype('category')

    for str_col in str_cols:
        train[str_col] = train[str_col].astype('float')
        test[str_col] = test[str_col].astype('float')

    return train, test


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    train = pd.read_json(train_file, lines=True)
    test = pd.read_json(test_file, lines=True)

    train, test = preprocessing_data(train, test)
    y = train['awards']
    train.drop(columns=['awards'], inplace=True)

    best_parameters = {'learning_rate': 0.05310671628228285, 'n_estimators': 206, 'max_depth': 10}
    regressor = LGBMRegressor(**best_parameters)
    regressor.fit(train, y)
    return regressor.predict(test)
