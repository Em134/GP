import numpy as np
import airsim


class UAV(object):
    """
    本类所用数据类型
    除获取无人机邻居为元组、无人机邻居集合为列表外
    其余均为np.array
    """
    def __init__(self, i, client, threshold=2, u=None, is_leader=False):
        """
        切换策略
        控制方式 hierarchical pattern:  1 分层;
        控制方式 egalitarian pattern:   2 平等;
        :param i:                   无人机编号;
        :param u:                   控制量输入;
        :param is_leader:           判断是否为首领机, 默认0号是;
        :param client               与ue4所连接的client;
        """
        self.dimension = 2
        self.velocity_scalar = None
        self.direction = None
        self.position = None
        self.threshold = threshold
        self.client = client
        self.ID = i
        self.input = u
        self.is_leader = is_leader
        self.pattern = 2
        self.patterns = {1: 'hierarchical', 2: 'egalitarian'}
        self.neighbors = None
        self.next_state = None
        self.step = 0

    # 控制量输入

    def pattern_switch(self):
        """
        1: 层级; 2: 平等;
        目前只有两类，所以用if语句实现简单切换
        :return:
        """
        if self.pattern == 1:
            self.pattern = 2
        else:
            self.pattern = 1

    def current_pattern(self):
        print("UAV", self.ID, "now is", self.patterns[self.pattern], self.pattern, "pattern.")
        return self.pattern

    def get_current_state(self):
        x = self.position[0]
        y = self.position[1]
        vel_x_self = self.direction[0] * self.velocity_scalar
        vel_y_self = self.direction[1] * self.velocity_scalar
        # 计算自己的速度

        return {"x": x, "y": y, "vx": vel_x_self, "vy": vel_y_self}

    def get_distance_with(self, target):
        state_self = self.get_current_state()
        state_target = target.get_current_state()

        distance = ((state_self['x'] - state_target['x']) ** 2 +
                    (state_self['y'] - state_target['y']) ** 2) / 1

        return distance

    def get_pos(self, pos):
        self.position = pos

    # 2D============================================================================================
    def consensus_based_2d(self, u_star, vehicles, alpha=0.3):  # 使无人机群目标地点一致
        """
        :param vehicles:    所有的无人机列表
        :param alpha:       自设参数，可自行选择合适的值或者选择n_i（与本无人机直接相邻的无人机的个数的倒数）;
        :param u_star:      期望速度，向量;
        :return:            返回控制无人机位置的输入控制量;
        """
        u = np.zeros((self.direction, 1))
        if self.is_leader:
            u = u_star + (u_star + 1) * 0.8 ** self.step
        else:
            pass

        neighbors = self.neighbors
        pos_self = np.array([[self.position[0][0]], [self.position[1][0]]])
        s = np.zeros((2, 1))

        for uav in neighbors:
            pos_neighbor = np.array([[uav.position[0][0]], [uav.position[1][0]]])
            s = s + pos_neighbor - pos_self

        u = u_star + alpha * s
        self.next_state = u
        return u

    def vicsek_model_2d(self, u_star, vehicles):  # 使无人机群方向一致
        """
        :param vehicles:    所有的载具
        :param u_star:      期望的速度, 标量;
        :return:            下一个时刻的控制无人机位置的输入控制量;
        """
        neighbors = self.neighbors
        s = np.zeros((2, 1))
        for uav in neighbors:
            # 还没分模式
            h = uav.consensus_based_2d(u_star, vehicles)
            s += h
        # u(k+1)
        # self.next_state = s / np.linalg.norm(s) * u_star
        # print(self.next_state)

    def get_neighbors_2d(self, vehicles):
        neighbors = []
        for v in vehicles:
            if self.get_distance_with(v) < self.threshold and v.ID != self.ID:
                neighbors.append(v)

        self.neighbors = neighbors
        return neighbors

    def update_state(self):
        self.step = self.step + 1
        print("position")
        print(self.position)
        self.position = self.position + self.next_state

    def set_v_scalar(self, v_scalar):
        self.velocity_scalar = v_scalar

    def set_direction(self, direction):
        self.direction = direction

