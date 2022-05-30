from FLB5 import UAV
from multi_uav import Group
from algorithm import Algorithm
import numpy as np
import matplotlib.pyplot as plt

times = 100
v_num = 5
vehicles = []
caner = [[[], []] for i in range(v_num)]
algo = Algorithm()
origin_x = [0, 10, 20, 10, 20, 20]
origin_y = [20, 20, 20, 10, 10, 0]
t = [i for i in range(times)]

for i in range(v_num):
    vehicles.append(UAV(i=i, position=np.array([[origin_x[i], [origin_y[i]]]])))

for step in range(times):
    v_cmd = algo.displacement_based(vehicles=vehicles)
    for i in range(v_num):
        caner[i][0].append(v_cmd[0, i])
        caner[i][1].append(v_cmd[1, i])

plt.plot(t, caner[1][0])
plt.plot(t, caner[1][1])
plt.show()
