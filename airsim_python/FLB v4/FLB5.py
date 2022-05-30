import numpy as np
import airsim


class UAV(object):
    """
    本类所用数据类型
    除获取无人机邻居为元组、无人机邻居集合为列表外
    其余均为np.array
    """
    def __init__(self, i, position, dimension=2, threshold=2, u=None, is_leader=False):
        """
        :param i:                   无人机编号;
        :param u:                   控制量输入;
        :param is_leader:           判断是否为首领机, 默认0号是;
        :param dimension:           无人机需要控制的维度：长、宽、高
        """
        self.dimension = dimension
        self.velocity_scalar = 2
        self.direction = np.array([[0] for i in range(self.dimension)])
        self.pos = position          # dimension * 1的np矩阵
        self.threshold = threshold
        self.ID = i
        self.input = u
        self.is_leader = is_leader
        self.step = 0
        self.vel = np.zeros((self.dimension, 1))  # 不单独给出方向和速度标量的话走这个

    def get_dir(self, next_dir):
        """
        将收到的方向信息格式转换为本类与Group类的通用格式，并调整为所需的维度
        :param next_dir: 将要朝向的位置
        :return:
        """
        if isinstance(next_dir, np.ndarray) and next_dir.shape == (self.dimension, 1):
            self.direction = next_dir

        elif isinstance(next_dir, list):
            temp = [[next_dir[i]] for i in next_dir]
            self.direction = np.array(temp)

        elif isinstance(next_dir, np.ndarray) and next_dir.shape == (self.dimension, ):
            temp = [[next_dir[i]] for i in range(next_dir.shape[0])]
            self.direction = np.array(temp)

    def get_vel(self, next_vel):
        if isinstance(next_vel, np.ndarray) and next_vel.shape == (self.dimension, 1):
            self.vel = next_vel

        elif isinstance(next_vel, list):
            temp = [[next_vel[i]] for i in next_vel]
            self.vel = np.array(temp)

        elif isinstance(next_vel, np.ndarray) and next_vel.shape == (self.dimension, ):
            temp = [[next_vel[i]] for i in range(next_vel.shape[0])]
            self.vel = np.array(temp)

    def get_pos(self, pos):
        """
        将收到的位置信息格式转换为本类与Group类的通用格式，并调整为所需的维度
        :param pos: 希望朝向的方向
        :return:
        """
        if pos.shape == (self.dimension, 1):
            self.pos = pos
        elif pos.shape == (3, ) or pos.shape == (2, ):
            temp = [[pos[i]] for i in pos]
            self.pos = np.array(temp)

    def output_pos(self):
        return self.pos

    def output_dir(self):
        return self.direction

    def update_state(self, state):
        """
        将group传递的状态字典中的信息更新给本无人机。
        :param state: 状态字典，该字典中包含了无人机的所有信息。
        :return:
        """
        temp_pos = [[state['x']], [state['y']], [state['z']]]
        temp_pos = temp_pos[0: self.dimension]
        temp_pos = np.array(temp_pos)
        self.pos = temp_pos

        # temp_vel = [[state['vx']], [state['vy']], [state['vz']]]
        # temp_vel = temp_vel[0: self.dimension]
        # temp_vel = np.array(temp_vel)
        # self.vel = temp_vel

    def show_vel(self):
        print('当前无人机：', self.ID, '速度为：', self.vel)

    def show_pos(self):
        print('当前无人机：', self.ID, '位置为：', self.pos)

    def name(self):
        return "UAV" + str(self.ID + 1)
