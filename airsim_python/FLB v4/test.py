from multi_uav import Group
import numpy as np
from algorithm import Algorithm
from time import *
A = np.array([[0, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 0],
              [0, 1, 0, 0, 1, 1],
              [0, 1, 0, 0, 1, 0],
              [0, 1, 1, 1, 0, 1],
              [0, 0, 1, 0, 1, 0]
              ])
total_time = 50

group = Group(v_num=6, neighbors_graph=A)
algo = Algorithm()
group.count_uav()
group.show_position()

group.connect_to_airsim()

group.iterate(algo.vision_based_small, total_time=50)
# group.iterate(algo.vision_based, total_time=50)
# group.iterate(algo.displacement_based, total_time=50)
# group.iterate(group.consensus_based, total_time=50)



