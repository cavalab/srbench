import numpy as np


def try_to_eliminate_constant(arr):
    sample_a = None
    for a in arr:
        if type(a) == np.ndarray and a.size != 1:
            sample_a = a
            break
    if sample_a is None:
        return False
    for i, a in enumerate(arr):
        if type(a) != np.ndarray or a.size == 1:
            arr[i] = np.full_like(sample_a, a)
    return True


def avg(*arr):
    arr = list(arr)
    if try_to_eliminate_constant(arr):
        return np.mean(np.stack(arr), axis=0)
    else:
        return np.mean(arr)


def max(*arr):
    arr = list(arr)
    if try_to_eliminate_constant(arr):
        return np.max(np.stack(arr), axis=0)
    else:
        return np.max(arr)


def min(*arr):
    arr = list(arr)
    if try_to_eliminate_constant(arr):
        return np.min(np.stack(arr), axis=0)
    else:
        return np.min(arr)


def add(*arr):
    sum = arr[0]
    for x in arr[1:]:
        sum = sum + x
    return sum


def sub(*arr):
    sum = arr[0]
    for x in arr[1:]:
        sum = sum - x
    return sum


threshold = 1e-6


def protect_divide(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > threshold, np.divide(x1, x2), 1.)


def analytical_quotient(x1, x2):
    return x1 / np.sqrt(1 + x2 ** 2)


def protect_sqrt(a):
    return np.sqrt(np.abs(a))
