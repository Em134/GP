import numpy as np
import matplotlib.pyplot as plt
from math import atan2

# 本函数迭代目标为：使得无人机的目标位置x趋于同一
# A矩阵表示每个无人机的邻居（平等模式），行代表无人机，列代表改行无人机的邻居是谁
A = np.array([[0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 0],
              [0, 1, 0, 0, 1, 1],
              [0, 1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, 0]
              ])
dt = 0.01
v_num = 6                                   # 无人机数量
u_star = 0.5                                # 期望速度，标量，直接与向量相加相当于在每个维度都加了这个标量
alpha = 0.2                                 # 公式中的alpha参数
total_time = 100                            # 总迭代时间

x = np.array([[i * 100 for i in range(v_num)],
              [0 for i in range(v_num)]])   # 初始化每个无人机的位置（只有x，y两个维度，没考虑z轴）

angle = [[] for i in range(v_num)]          # 用于存储每个无人机角度的变化
u_i = np.zeros((2, v_num))                  # 用于存储所有无人机迭代x用的控制量输出
t = [i for i in range(total_time)]          # 用于画图作为时间轴x轴
u_state = u_i.T

for step in range(total_time):              # 迭代开始
    # ++++++++++++++++++++++++++++++++++++++++++++++++画图用 开始
    temp = np.array([u_i[:, 0]]).T
    a = np.array([u_i[:, 0]]).T
    for j in range(v_num - 1):
        a = np.append(a, temp, axis=1)
    u_state = np.append(u_state, u_i.T - a.T, axis=1)
    print('u', u_state.shape)
    # ++++++++++++++++++++++++++++++++++++++++++++++++画图用 结束

    if step < 20:
        u_i[:, 0] = np.array([[1], [2]])[:, 0] * np.exp(-step * dt)
    else:
        u_i[:, 0] = u_star * np.array([[2], [2]])[:, 0] / 2 ** 0.5
    for i in range(1, v_num):                           # 对于第i个非首领无人机
        s = np.zeros((2, 1))                            # 用于计算求和部分

        for j in range(v_num):                          # i与第j个的x的差值的和
            s = s + u_i[:, j] * A[j, i]

        u_i[:, i] = (s / np.linalg.norm(s) * np.linalg.norm(u_i[:, 0]))[:, 0]  # u_i[:, 0]相当于公式中的u_star
    x = x + u_i                                         # 对无人机位置进行迭代

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++ 画图用开始
for y in u_state:
    y = [y[i] for i in range(0, len(y)-2, 2)]
    plt.plot(t, y)
plt.legend(['uav0', 'uav1', 'uav2', 'uav3', 'uav4', 'uav5'])
plt.show()
