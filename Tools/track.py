import numpy as np
from matplotlib import pyplot as plt

def track_seg(points, color='r'):
    for p in points:
        plt.plot(p[0], p[1])
    
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        plt.quiver(points[i][0], points[i][1], dx, dy, color=color, angles='xy', scale=1.03, scale_units='xy', width=0.005)

if __name__ == '__main__':
    colors = ['r', 'b', 'm', 'g']
    base = np.linspace(-3.14, 3.14, 128)
    points = np.stack((np.cos(base), np.sin(base))).T.reshape(4, -1, 2)
    
    for i, seg in enumerate(points):
        track_seg(seg, colors[i])

    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
