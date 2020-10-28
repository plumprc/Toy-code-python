import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 设置画布
    fig = plt.figure()
    plt.xlim(-10, 90)
    plt.ylim(-10, 100)

    points = [[0, 8], [26, 43], [41, 60], [50, 66], [68, 87]]

    for p in points:
        # plt.plot(p.x, p.y, "o")
        plt.plot(p[0], p[1])
    
    # 画箭头
    for i in range(len(points) - 1):
        dx = points[i+1][0] - points[i][0]
        dy = points[i+1][1] - points[i][1]
        plt.quiver(points[i][0], points[i][1], dx, dy, angles='xy', scale=1.03, scale_units='xy', width=0.005, color='r')

    plt.show()
