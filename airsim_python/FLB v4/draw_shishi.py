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

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
ay2 = []
ay3 = []
ay4 = []
ay5 = []

plt.ion()  # 开启一个画图的窗口
k = 6  # k对应label0
for i in range(100):  # 遍历0-99的值
    ax.append(i)  # 添加 i 到 x 轴的数据中
    ay.append(data[0][:, k][i])  # 添加 i 的平方到 y 轴的数据中
    ay2.append(data[1][:, k][i])
    ay3.append(data[2][:, k][i])
    ay4.append(data[3][:, k][i])
    ay5.append(data[4][:, k][i])
    plt.clf()  # 清除之前画的图
    plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.plot(ax, ay2)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.plot(ax, ay3)
    plt.plot(ax, ay4)
    plt.plot(ax, ay5)
    plt.pause(0.1)  # 暂停一秒
    # plt.ioff()  # 关闭画图的窗口


