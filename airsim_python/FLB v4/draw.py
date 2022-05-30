import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = []
for i in range(5):
    data.append(np.loadtxt(open("./data/vision_based_normal/uav" + str(i) + ".csv", "rb"), delimiter=",", skiprows=1))

t = [i for i in range(data[0].shape[0])]

labels = ['position x', 'position y', 'position z',
          'velocity x', 'velocity y', 'velocity z',
          'acceleration x', 'acceleration y', 'acceleration z',
          'pitch', 'roll', 'yaw',
          'pitch_a', 'roll_a', 'yaw_a']

fig, a = plt.subplots(3, 3)
k = 0
for i in range(3):  # 子图横坐标
    for j in range(3):  # 子图纵坐标
        legends = []
        for c in range(5):
            a[i][j].plot(t, data[c][:, k + 2])
            a[i][j].set_ylabel(labels[k])
            a[i][j].set_xlabel('time')
            legends.append('uav'+str(c))
        a[i][j].legend(legends)
        k += 1

plt.show()

"""每三幅图"""
# fig, a = plt.subplots(1, 3)
# k = 0
# for i in range(1):  # 子图横坐标
#     for j in range(3):  # 子图纵坐标
#         legends = []
#         for c in range(5):
#             a[j].plot(t, data[c][:, k + 1 + 3])  # 前三个数据则 +2
#             a[j].set_ylabel(labels[k + 3])
#             a[j].set_xlabel('time')
#             legends.append('uav'+str(c))
#         a[j].legend(legends)
#         k += 1
#
# plt.show()

