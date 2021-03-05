import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = list()
y = list()
zs1 = list()
zs2 = list()
zs3 = list()


fh = open('./results.txt', 'r')
for line in fh:
    line = line.replace('(', '')
    line = line.replace(')', '')
    line = line.replace('\n', '')
    line = line.split(',')
    y.append(float(line[0]))
    x.append(float(line[1]))
    zs1.append(float(line[2]))
    zs2.append(float(line[2]) + float(line[3]))
    zs3.append(float(line[2]) + float(line[3]) + float(line[4]))


x = np.array(x)
y = np.array(y)
zs1 = np.array(zs1)
zs2 = np.array(zs2)
zs3 = np.array(zs3)

X, Y = np.meshgrid(x, y)
Z1, Z1_aux = np.meshgrid(zs1, zs1)
Z2, Z2_aux = np.meshgrid(zs2, zs2)
Z3, Z3_aux = np.meshgrid(zs3, zs3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, Z1)
# ax.plot_surface(X, Y, Z2)
# ax.plot_surface(X, Y, Z3)
ax.scatter(x, y, zs1)
ax.scatter(x, y, zs2)
ax.scatter(x, y, zs3)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
