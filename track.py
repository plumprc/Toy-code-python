import numpy as np
from matplotlib import pyplot as plt

sketch = np.load('lab/DataSets/dog.npy', allow_pickle=True).tolist()
# points = [[0, 8], [26, 43], [41, 60], [50, 66], [68, 87]]
colors = ['r', 'chocolate', 'orange', 'y', 'lightseagreen', 
        'deepskyblue', 'g', 'b', 'm']

def track_seg(points, color='r'):
    for p in points:
        plt.plot(p[0], p[1])
    
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        plt.quiver(points[i][0], points[i][1], dx, dy, color=color, angles='xy', scale=1.03, scale_units='xy', width=0.005)

if __name__ == '__main__':
    for i, seg in enumerate(sketch):
        track_seg(seg, colors[i])
    # track_seg(points)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.show()
