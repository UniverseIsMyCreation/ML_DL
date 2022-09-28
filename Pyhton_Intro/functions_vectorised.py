import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    X = np.diag(X)
    return np.sum(X[X>=0]) if len(X[X>=0]) else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x = np.sort(x)
    y = np.sort(y)
    return (x == y).all()


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    result = x[1:] * x[:x.shape[0]-1]
    return np.amax(result[result % 3 == 0]) if len(result[result % 3 == 0]) else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения c указанными весами.
    """
    res_image = np.zeros(shape = (image.shape[:2]))
    for i in range(len(weights)):
        res_image += image[:,:,i] * weights[i]
    
    return res_image


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    new_x = np.repeat(x.ravel()[::2],x.ravel()[1::2])
    new_y = np.repeat(y.ravel()[::2],y.ravel()[1::2])
    if len(new_x) != len(new_y):
        return -1
    
    return np.dot(new_x,new_y)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    scalar_XY = np.matmul(X,Y.T)
    len_x = np.sum(X*X, axis = 1) ** 0.5
    len_y = np.sum(Y*Y, axis = 1) ** 0.5
    len_x = len_x.reshape(len(len_x),1)
    len_y = len_y.reshape(1,len(len_y))
    div = np.matmul(len_x,len_y) 
    zero_len = div < 0.000005
    scalar_XY[zero_len] = 1
    not_zero_len = div >= 0.000005
    scalar_XY[not_zero_len] = scalar_XY[not_zero_len] / div[not_zero_len]

    return scalar_XY
