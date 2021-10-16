import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
# from mayavi import mlab
from mpl_toolkits.mplot3d import Axes3D


x = list()
y = list()
zs1 = list()
zs2 = list()
zs3 = list()


fh = open('./results.txt', 'r')
points = list()
for line in fh:
    line = line.replace('(', '')
    line = line.replace(')', '')
    line = line.replace('\n', '')
    line = line.split(',')
    if float(line[1]) >= 15.0:
        continue
    line = (float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]))
    points.append(line)


points = np.array(points)
points_sort_idx = np.lexsort((points[:, 1], points[:, 0]))
points = points[points_sort_idx]


x = np.flip(points[:, 1])
y = np.flip(points[:, 0])
zs1 = points[:, 2] * 1e6
zs2 = points[:, 3] * 1e6 + zs1
zs3 = points[:, 4] * 1e6 + zs2

zs1 = ndimage.uniform_filter(zs1, size=3)
zs2 = ndimage.uniform_filter(zs2, size=3)
zs3 = ndimage.uniform_filter(zs3, size=3)

# print(int(int(np.amax(x) - np.amin(x))) + 1)
# quit()
# zs_aux = zs1.reshape((int(np.amax(y) - np.amin(y)) + 1, int(np.amax(x) - np.amin(x)) + 1))
zs_aux = zs1.reshape((int(np.amax(y) - np.amin(y)) + 1, int(15.0 - 0.0)))
avg_wgt = np.zeros(zs_aux.shape)
idxX = zs_aux.shape[0]
idxY = zs_aux.shape[1]
sq_ones_size = 10
sq_ones = np.ones((sq_ones_size, sq_ones_size))
mid_idxX = int((idxX - sq_ones_size)/2)
mid_idxY = int((idxY - sq_ones_size)/2)
avg_wgt[mid_idxX:mid_idxX + sq_ones_size, mid_idxY:mid_idxY + sq_ones_size] = sq_ones
avg_wgt = avg_wgt.reshape(y.size)


z1_val = np.average(zs1, weights=avg_wgt)
z2_val = np.average(zs2, weights=avg_wgt)
z3_val = np.average(zs3, weights=avg_wgt) - z2_val
z2_val = z2_val - z1_val

print('Capa 1 =', z1_val, 'um')
print('Capa 2 =', z2_val, 'um')
print('Capa 3 =', z3_val, 'um')


# X, Y = np.meshgrid(x, y)
# Z1, Z1_aux = np.meshgrid(zs1, zs1)
# Z2, Z2_aux = np.meshgrid(zs2, zs2)
# Z3, Z3_aux = np.meshgrid(zs3, zs3)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax = fig.gca(projection='3d')
# ax.set_spect([1.0, 1.6, 0.25])

# ax.plot_surface(X, Y, Z1, color='tab:blue')
# ax.plot_surface(X, Y, Z2, color='tab:orange')
# ax.plot_surface(X, Y, Z3, color='tab:green')
ax.plot_trisurf(np.flip(x), y, zs1, color='tab:blue')
ax.plot_trisurf(np.flip(x), y, zs2, color='tab:orange')
ax.plot_trisurf(np.flip(x), y, zs3, color='tab:green')
# ax.scatter(x, y, zs1, color='tab:blue')
# ax.scatter(x, y, zs2, color='tab:orange')
# ax.scatter(x, y, zs3, color='tab:green')

#### Tweaks para el plot ####
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel(r'Z ($\mu m$)')

# ax.set_xlim3d(15, 19)
# ax.set_ylim3d(33, 37)
ax.set_zlim3d(0, 350)

plt.show()
