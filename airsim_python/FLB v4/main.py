from multi_uav import Group
import numpy as np
from algorithm import Algorithm

A = np.array([[0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 0],
              [0, 1, 0, 0, 1, 1],
              [0, 1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, 0]
              ])
total_time = 50

group = Group(v_num=5, neighbors_graph=A)
algo = Algorithm()
group.count_uav()
group.show_position()

# group.connect_to_airsim()
# group.iterate(algo.vision_based_cb, total_time=100)
group.iterate(group.consensus_based, total_time=50)



