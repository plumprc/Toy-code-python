import numpy as np, matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.pyplot import MultipleLocator

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
y = np.array([0, 0, 32, 23, 20, 48, 53, 77, 12, 9, 12, 12])
x_major_locator = MultipleLocator(1)

# 平滑曲线，有点问题
# x_m = np.linspace(x.min(), x.max(), 150)
# y_m = make_interp_spline(x, y)(x_m)

plt.xlim(0, 13)
plt.ylim(0, 80)
# plt.plot(x_m, y_m)
# plt.grid()
plt.bar(x, y)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)

plt.title('Tasks', fontsize=13)
plt.xlabel('Month', fontsize=12)
plt.ylabel('#Num', fontsize=12)

plt.show()