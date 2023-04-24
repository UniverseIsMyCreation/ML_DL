import numpy as np

import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        '''
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        '''
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, data, labels):
        '''
        Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        '''
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init="auto",
            max_iter=750
        ).fit(data)
        self.cluster_labels = self.kmeans.labels_
        self.ground_truth_labels = labels
        self.mapping, self.predicted_labels = self._best_fit_classification(
            self.cluster_labels,
            self.ground_truth_labels
        )

        return self

    def predict(self, data):
        '''
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        '''
        data_labels = np.array(self.kmeans.predict(data))
        preds = np.zeros_like(data_labels)
        unique_labels = np.unique(data_labels)
        for unique_label in unique_labels:
            mask = data_labels == unique_label
            pred_cls = self.mapping[unique_label]
            preds[mask] = pred_cls

        return preds

    def _best_fit_classification(self, cluster_labels, true_labels):
        '''
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        '''
        true_labels_np = np.array(true_labels)
        cluster_labels_np = np.array(cluster_labels)
        unique_labels, label_counts = np.unique(true_labels_np[true_labels_np != -1], return_counts=True)
        max_amount_label = unique_labels[np.argmax(label_counts)]
        all_cluster_labels_np = np.unique(cluster_labels_np)
        mapping = np.full(shape=(self.n_clusters), fill_value=-1, dtype=np.int64)
        for cluster_label in all_cluster_labels_np:
            mask = cluster_labels_np == cluster_label
            true_cluster_labels = true_labels_np[mask]
            if len(true_cluster_labels[true_cluster_labels != -1]) == 0:
                cluster_label_predicted = max_amount_label
            else:
                unique, counts = np.unique(true_cluster_labels[true_cluster_labels != -1], return_counts=True)
                cluster_label_predicted = unique[np.argmax(counts)]
            true_labels_np[mask] = cluster_label_predicted
            mapping[cluster_label] = cluster_label_predicted

        mapping[mapping == -1] = max_amount_label

        return mapping, true_labels_np
