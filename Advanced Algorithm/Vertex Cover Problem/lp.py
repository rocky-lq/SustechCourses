import time

import numpy as np
import scipy.io as io
from scipy.optimize import linprog


def sample_1():
    A = np.array([1, 1])
    B = np.array([[1, 2], [2, 1]])
    C = np.array([10, 8]).transpose()

    m, n = 2, 2
    bounds = np.array([[0, None] for _ in range(n)])
    start_time = time.time()
    res = linprog(np.dot(A, -1), B, C, A_eq=None, b_eq=None, method=method, bounds=bounds)
    end_time = time.time()
    print('sample_1')
    print(res.x)
    print(f'result: {np.dot(A, res.x)}')
    print("cost time: {:6f}s".format(end_time - start_time))
    print()
    ...


def sample_2():
    A = np.array([2, 1])
    B = np.array([[1, 2], [2, 1]])
    C = np.array([10, 8]).transpose()

    m, n = 2, 2
    bounds = np.array([[0, None] for _ in range(n)])
    start_time = time.time()
    res = linprog(np.dot(A, -1), B, C, A_eq=None, b_eq=None, method=method, bounds=bounds)
    end_time = time.time()
    print('sample_2')
    print(res.x)
    print(f'result: {np.dot(A, res.x)}')
    print("cost time: {:6f}s".format(end_time - start_time))
    print()
    ...


def sample_3():
    A = np.array([0, 0, 2])
    B = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 1]])
    C = np.array([2, -1, 2]).transpose()

    m, n = 3, 3
    bounds = np.array([[0, None] for _ in range(n)])
    start_time = time.time()
    res = linprog(np.dot(A, -1), B, C, A_eq=None, b_eq=None, method=method, bounds=bounds)
    end_time = time.time()
    print('sample_3')
    print(res.x)
    print(f'result: {np.dot(A, res.x)}')
    print("cost time: {:6f}s".format(end_time - start_time))
    print()
    ...


def solve(data, title):
    A = data['A']
    B = data['B']
    C = data['C']
    m, n = B.shape

    bounds = np.array([[0, None] for _ in range(n)])

    start_time = time.time()
    res = linprog(np.dot(A, -1), B, C, A_eq=None, b_eq=None, method=method, bounds=bounds)
    end_time = time.time()
    print(title)
    print(f'result: {np.dot(A, res.x)[0]}')
    print("cost time: {:.6f}s".format(end_time - start_time))
    print()
    ...


def sample_test():
    A = np.array([1, 1, 1, 1, 1])
    B = np.array([[-1, -1, 0, 0, 0], [0, -1, -1, 0, 0], [0, 0, -1, -1, 0], [0, 0, 0, -1, -1], [-1, 0, 0, 0, -1]])
    C = np.array([-1, -1, -1, -1, -1]).transpose()

    m, n = B.shape
    bounds = np.array([[0, 1] for _ in range(n)])
    start_time = time.time()
    res = linprog(np.dot(A, -1), A_ub=B, b_ub=C, A_eq=None, b_eq=None, method=method, bounds=bounds)
    end_time = time.time()
    print('sample_test')
    print(res.x)
    print(f'result: {np.dot(A, res.x)}')
    print("cost time: {:6f}s".format(end_time - start_time))
    print()
    ...


# Mehod = 'highs-ipm', 'highs-ds' None
if __name__ == '__main__':
    method = 'highs-ipm'
    # print(method)
    # print()
    # data = io.loadmat('instance_small.mat')
    # solve(data, 'instance_small')
    #
    # data = io.loadmat('instance_medium.mat')
    # solve(data, 'instance_medium')
    #
    # data = io.loadmat('instance_large.mat')
    # solve(data, 'instance_large')
    # sample_1()
    # sample_2()
    # sample_3()
    sample_test()
    ...
