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



