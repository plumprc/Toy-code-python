import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

if __name__ == '__main__':
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    y = np.array([0, 0, 32, 23, 20, 48, 53, 77, 12, 9, 12, 12])
    x_major_locator = MultipleLocator(1)

    plt.xlim(0, 13)
    plt.ylim(0, 80)
    plt.bar(x, y)

    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.title('Tasks', fontsize=13)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('#Num', fontsize=12)

    plt.show()
