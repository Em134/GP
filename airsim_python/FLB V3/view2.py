import matplotlib.pyplot as plt
from FLB4 import UAV
import numpy as np
from math import atan2

client = None
v_scalar = 0.3
vehicle_nums = 10
vehicles = [UAV(i, client) for i in range(vehicle_nums)]


def init_vehicles(uav):
    for v in uav:
        v.direction = np.random.rand(2, 1)
        v.velocity_scalar = v_scalar
        v.position = np.array([2 * np.random.random(2)]).T


init_vehicles(vehicles)
for v in vehicles:
    v.get_neighbors_2d(vehicles)


# 存每个无人机的角度
steps = 100
v_dis = [[] for i in range(vehicle_nums)]
v_pos = [[] for i in range(vehicle_nums)]
times = [i for i in range(steps)]

for t in range(steps):
    for v in vehicles:
        print('当前无人机：', v.ID, '轮数：', t, '\n')
        angle = atan2(v.direction[0][0], v.direction[1][0])
        # plt.scatter(t, angle)
        v.get_neighbors_2d(vehicles)
        v.consensus_based_2d(v_scalar, vehicles)
        v.update_state()
        v_dis[v.ID].append(np.linalg.norm(vehicles[0].position - v.position))


for i in range(vehicle_nums):
    plt.plot(times, v_dis[i])
plt.show()


