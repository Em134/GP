from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
# 创建绘图区域
ax = plt.axes(projection='3d')
# 构建xyz
z = np.linspace(0, 1, 100)
x = z * np.sin(20 * z)
y = z * np.cos(20 * z)
c = x + y

x = np.array([1 * 0.1 for i in range(10)])
y = np.array([1 for i in range(10)])
z = np.array([i * 0.1 for i in range(10)])

ax.scatter3D(x, y, z)
ax.set_title('3d Scatter plot')
plt.show()

