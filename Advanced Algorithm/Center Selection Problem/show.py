import matplotlib.pyplot as plt
import numpy as np

alginfo = {0: "Center Selection",
           1: "Distance-based Greedy Removal",
           2: "Greedy Inclusion",
           3: "Greedy Removal"}


class Tuple:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def show_centers(tuples, centers, remove, ax, title):
    x = [tuple.x for tuple in tuples]
    y = [tuple.y for tuple in tuples]
    ax.scatter(x, y, s=50, lw=0, cmap='RdYlGn', color='g')

    for i in range(len(centers)):
        ax.scatter(tuples[centers[i]].x, tuples[centers[i]].y, s=80, lw=0, cmap='RdYlGn', color='r')
        if len(remove) == 0:
            ax.annotate(i + 1, xy=(tuples[centers[i]].x - 0.3, tuples[centers[i]].y + 0.3))

    for i in range(len(remove)):
        ax.scatter(tuples[remove[i]].x, tuples[remove[i]].y, s=80, lw=0, cmap='RdYlGn', color='b')
        ax.annotate(i + 1, xy=(tuples[remove[i]].x - 0.3, tuples[remove[i]].y + 0.3))

    ax.axis(xmin=0, xmax=8, ymin=0, ymax=8)
    ax.grid(True)
    ax.set_title(title)
    ax.set_aspect(1)


if __name__ == '__main__':
    tuples = []
    centers = [[] for i in range(4)]
    removes = [[] for i in range(4)]
    with open('alg_compare_data28.txt', 'r') as fr:
        info = fr.readline().split(' ')
        [n, k] = list(map(int, info))
        print(n, k)
        for i in range(n):
            info = fr.readline().split(' ')
            [order, x, y] = list(map(float, info))
            tuple = Tuple(x, y)
            tuples.append(tuple)
        for i in range(4):
            info = fr.readline().split(' ')
            center = list(map(int, info))
            centers[i] = center

            info = fr.readline().split(' ')
            if info[0] != '\n':
                remove = list(map(int, info))
            else:
                remove = []
            removes[i] = remove

    print(removes)
    fig = plt.figure(figsize=(10, 10))
    fig_row = 2
    fig_col = 2

    for i in range(4):
        ax = fig.add_subplot(fig_row, fig_col, i + 1)
        show_centers(tuples, centers[i], removes[i], ax, alginfo[i])
    print(centers)

    # plt.show()
    plt.savefig(f'test{n}.png', bbox_inches='tight', dpi=300)

    # print(x, y)
    # x = range(1, 10)
    # y = range(1, 10)
    # plt.plot(x, y)
    # plt.show()
