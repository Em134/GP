import numpy as np
import matplotlib.pyplot as plt

# 本函数迭代目标为：使得无人机的目标位置x趋于同一
# A矩阵表示每个无人机的邻居（平等模式），行代表无人机，列代表改行无人机的邻居是谁
A = np.array([[0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 0],
              [0, 1, 0, 0, 1, 1],
              [0, 1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, 0]
              ])
v_num = 6                                   # 无人机数量
u_star = 0.5                                # 期望速度，标量，直接与向量相加相当于在每个维度都加了这个标量
alpha = 0.3                                 # 公式中的alpha参数
total_time = 100                            # 总迭代时间

x = np.array([[np.random.rand() for i in range(v_num)],
              [np.random.rand() for i in range(v_num)]])   # 初始化每个无人机的位置（只有x，y两个维度，没考虑z轴）

distance = [[] for i in range(v_num)]       # 用于存储每个无人机与首领无人机的距离
u_i = np.zeros((2, v_num))                  # 用于存储所有无人机迭代x用的控制量输出
t = [i for i in range(total_time)]          # 用于画图作为时间轴x轴

for step in range(total_time):              # 迭代开始
    # ++++++++++++++++++++++++++++++++++++++++++++++++画图用 开始
    for i in range(v_num):
        print(x[:, 0], x[:, i], np.linalg.norm(x[:, 0] - x[:, i]))
        distance[i].append(np.linalg.norm(x[:, 0] - x[:, i]))
    # ++++++++++++++++++++++++++++++++++++++++++++++++画图用 结束

    u_i[:, 0] = u_star + (u_star + 1) * 0.8 ** step     # 对于首领无人机直接更新
    for i in range(1, v_num):                           # 对于第i个非首领无人机
        s = np.zeros((2, 1))                            # 用于计算求和部分

        for j in range(v_num):                          # i与第j个的x的差值的和
            s = s + np.array([x[:, j] - x[:, i]]).T * A[j, i]

        u_i[:, i] = (s * alpha + u_star)[:, 0]
    x = x + u_i                                         # 对无人机位置进行迭代

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++ 画图用开始
for y in distance:
    plt.plot(t, y)
plt.legend(['uav0', 'uav1', 'uav2', 'uav3', 'uav4', 'uav5'])
plt.ylabel('distance between UAV0 and others')
plt.show()
