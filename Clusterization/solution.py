import numpy as np

from sklearn.metrics import pairwise_distances


def silhouette_score(x, labels):
    '''
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''

    sil_sum = 0
    sample_volume = len(x)
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        return 0

    x_counts = dict()
    statistics_amount = np.bincount(labels)
    mask_take = statistics_amount > 0
    statistics_amount = statistics_amount[statistics_amount != 0]

    for idx, label in enumerate(unique_labels):
        x_counts[label] = idx

    distances = pairwise_distances(x, x, metric='euclidean')
    for idx, obj in enumerate(distances):
        obj_label = labels[idx]
        obj_mask = labels == obj_label
        cluster_pos = x_counts[obj_label]
        if np.sum(obj_mask) == 1:
            continue
        statistics_distances = np.bincount(labels, obj)[mask_take]
        cur_cluster_dist = statistics_distances[cluster_pos]
        cur_cluster_amount = statistics_amount[cluster_pos]
        s = cur_cluster_dist / (cur_cluster_amount - 1)

        rest_statistics_distances = np.concatenate((statistics_distances[:cluster_pos], statistics_distances[cluster_pos + 1:]))
        rest_statistics_amount = np.concatenate((statistics_amount[:cluster_pos], statistics_amount[cluster_pos + 1:]))
        d = np.min(rest_statistics_distances / rest_statistics_amount)

        if s == d == 0:
            continue
        else:
            sil_sum += ((d - s) / max(d, s))

    return sil_sum / sample_volume


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''

    if len(true_labels) == 1:
        return 1.0

    precision = list()
    for i, cluster_num in enumerate(predicted_labels):
        true_label = true_labels[i]
        real_clusters = true_labels[cluster_num == predicted_labels]
        res = np.where(real_clusters != true_label, 0, 1)
        precision.append(np.mean(res))
    precision = np.mean(precision)

    recall = list()
    for i, cluster_num in enumerate(true_labels):
        predicted_label = predicted_labels[i]
        predicted_clusters = predicted_labels[cluster_num == true_labels]
        res = np.where(predicted_clusters != predicted_label, 0, 1)
        recall.append(np.mean(res))
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)

    return f1
