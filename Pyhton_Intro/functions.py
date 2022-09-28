from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    amount = 0
    sum = 0
    for i in range(min(len(X[0]),len(X))):
        if X[i][i]>= 0:
            sum += X[i][i]
            amount += 1
    return sum if amount else -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x = sorted(x)
    y = sorted(y)
    return x == y


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    max = 0
    flag = False
    for i in range(len(x)-1):
        if x[i] % 3 == 0 or x[i+1] % 3 == 0:
            if not flag:
                max = x[i] * x[i+1]
                flag = True
            elif x[i] * x[i+1] > max:
                max = x[i] * x[i+1]
    return max if flag else -1



def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res_image = []
    for i in range(len(image)):
        res_image.append([0 for i in range(len(image[0]))])
    for i in range(len(image)):
        for j in range(len(image[0])):
            for num_image in range(len(weights)):
                res_image[i][j] += image[i][j][num_image] * weights[num_image]

    return res_image

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    len_x, len_y = 0,0
    new_x,new_y = [],[]
    for i in range(len(x)):
        len_x += x[i][1]
        for j in range(x[i][1]):
            new_x.append(x[i][0])
    for i in range(len(y)):
        len_y += y[i][1]
        for j in range(y[i][1]):
            new_y.append(y[i][0])
    if len_x != len_y:
        return -1
    result = 0
    for i in range(len_x):
        result +=  new_x[i] * new_y[i]
    
    return result

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    result = []
    for i in range(len(X)):
        result.append([0 for j in range(len(Y))])
    for i in range(len(X)):
        for j in range(len(Y)):
            scalar_i_j = 0
            len_x,len_y = 0,0
            for k in range(len(X[0])):
                scalar_i_j += X[i][k] * Y[j][k]
                len_x += X[i][k] ** 2
                len_y += Y[j][k] ** 2
            len_x = len_x ** 0.5
            len_y = len_y ** 0.5
            if len_x < 0.000005 or len_y < 0.000005:
                scalar_i_j = 1
            else:
                scalar_i_j /= len_x
                scalar_i_j /= len_y
            result[i][j] = scalar_i_j
    
    return result
