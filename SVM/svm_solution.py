import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):

    scaler = StandardScaler()
    model = SVC(kernel='rbf', C=50, class_weight={1: 0.5, 0: 0.5})

    scaler.fit_transform(train_features)
    scaler.transform(test_features)
    model.fit(train_features, train_target)

    return model.predict(test_features)