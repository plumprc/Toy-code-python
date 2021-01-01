import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

x = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = np.array([32, 23, 20, 48, 53, 77, 12, 9, 12, 12])

x_m = np.linspace(x.min(), x.max(), 150)
y_m = make_interp_spline(x, y)(x_m)

plt.scatter(x, y)
plt.plot(x_m, y_m)

plt.show()