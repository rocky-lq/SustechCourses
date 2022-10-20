import matplotlib.pyplot as plt
import numpy as np


class Tuple:
    def __init__(self, x, y):
        self.x = x
        self.y = y


if __name__ == '__main__':
    tuples = []
    with open('in.txt', 'r') as fr:
        info = fr.readline().split(' ')
        [n, k] = list(map(int, info))
        print(n, k)
        for i in range(n):
            info = fr.readline().split(' ')
            [x, y] = list(map(float, info))
            tuple = Tuple(x, y)
            tuples.append(tuple)

    x = [tuple.x for tuple in tuples]
    y = [tuple.y for tuple in tuples]
    plt.scatter(x, y, s=50, lw=0, cmap='RdYlGn', color='g')
    plt.grid(True)
    plt.show()

    # print(x, y)
    # x = range(1, 10)
    # y = range(1, 10)
    # plt.plot(x, y)
    # plt.show()
