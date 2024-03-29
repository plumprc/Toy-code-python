import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # y = np.array([23, 25, 26, 1, 14, 4, 6, 13, 22, 40, 23, 26])
    # y = np.array([13, 12, 18, 35, 25, 27, 43, 20, 32, 19, 29, 23])
    y = np.array([29, 16, 25, 18, 45, 39, 47, 67, 22, 33, 23, 6])
    x_major_locator = MultipleLocator(1)
    y_max = np.max(y) + 2

    plt.xlim(0, 13)
    plt.ylim(0, y_max)
    plt.bar(x, y)

    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.title('Tasks', fontsize=13)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('#Num', fontsize=12)

    plt.show()
