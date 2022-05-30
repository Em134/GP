from FLB5 import UAV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import airsim
import math
import time
from csv_test import Data2csv


class Group(object):
    def __init__(self, v_num, dimension=2, neighbors_graph=None, init_p=None):
        """
        无人机群类，用于对群体的位置进行迭代
        :param v_num:                   本群体的无人机的数量
        :param client:                  本群体的airsim通信句柄
        :param neighbors_graph:         本群体的邻居关系
        :param init_pos:                初试每个无人机的位置信息
        """

        self.threshold = 5  # 不用图来判断邻居的时候，判断是否为邻居的距离阈值
        self.dimension = dimension  # 无人机所需要的维度（长、宽、高）
        self.v_num = v_num  # 无人机数量
        self.client = None  # airsim通信句柄
        self.neighbors_graph = neighbors_graph  # 表示邻居关系的矩阵
        self.vehicles = []  # 无人机群的所有无人机
        self.pattern = 0  # 无人机群控制模式
        self.pos = self.init_pos(init_p)  # self.pos存储的是无人机接下来要前往的位置
        self.u = np.zeros((self.dimension, self.v_num))  # 用于存储所有无人机迭代x用的控制量输出

        # 画图用
        self.distance = [[] for i in range(self.v_num)]  # 用于存储每个无人机与首领无人机的距离

        self.__init_uav()

    def consensus_based(self, step, alpha=0.3, u_star=0.5):
        """
        模式0：层级模式，有leader，默认为0号机。
        模式1：pass
        没运行本函数一次，迭代参数一次
        :return: 无
        """
        if not self.is_connected():  # 如果没有client说明没接入airsim，采用画图仿真
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 画图用 开始
            for i in range(self.v_num):
                print(self.pos[:, 0], self.pos[:, i], np.linalg.norm(self.pos[:, 0] - self.pos[:, i]))
                self.distance[i].append(np.linalg.norm(self.pos[:, 0] - self.pos[:, i]))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 画图用 结束

            if self.pattern == 0:
                self.u[:, 0] = u_star + (u_star + 1) * 0.8 ** step  # 对于首领无人机直接更新
                for i in range(1, self.v_num):  # 对于第i个非首领无人机
                    s = np.zeros((self.dimension, 1))  # 用于计算求和部分

                    for j in range(self.v_num):  # i与第j个的x的差值的和
                        s = s + np.array([self.pos[:, j] - self.pos[:, i]]).T * self.neighbors_graph[j, i]

                    self.u[:, i] = (s * alpha + u_star)[:, 0]
                self.pos = self.pos + self.u  # 对无人机位置进行迭代

            else:  # 连接到airsim的情况
                pass

    def get_airsim_current_total_pos_array(self, state_list):
        """
        根据get_i_state中获取的字典，将整个无人机群的目标维度内的位置信息收集为矩阵
        :param state_list:  get_i_state中获取的字典
        :return:            无人机群位置信息矩阵
        """
        pos_list = []
        for state in state_list:
            x = [state['x']]
            y = [state['y']]
            z = [state['z']]
            pos = [x, y, z]
            pos = pos[0: self.dimension]
            pos = np.array(pos).T
            pos_list.append(pos)
        pos_array = np.array(pos_list)
        return pos_array

    # 道具=================================================================================================
    def __init_uav(self):
        """
        初始化无人机群的每一个无人机
        初始化每个无人机的位置
        x：dimension * v_num的矩阵
        例如二维每个无人机的位置信息为一个2*1尺寸的列向量，类型为np.array：x_i = np.array([[a], [b]])
        :return: 无
        """
        for i in range(self.v_num):
            self.vehicles.append(UAV(i=i, position=np.array([self.pos[:, i]]).T))

    def init_pos(self, init_p):
        """
        初始化无人机位置信息
        :param init_p: 未连接airsim的情况下，有可能收到的初试位置参数
        :return: 无
        """
        if init_p is None:  # 如果初始化无人机群时，没有给出无人机的初始位置，则随机给出
            # return np.array([[0, 100, 200, 300, 400], [0, 0, 0, 0, 0]])
            return np.array([[np.random.rand() for i in range(self.v_num)] for j in range(self.dimension)])
        else:
            return init_p

    def set_group_pos(self):
        """
        在连接到airsim后才生效，用于设定无人机群
        :return: 无
        """
        if self.is_connected():
            state_list = []
            for i in range(self.v_num):  # 获得所有无人机的信息字典
                state_list.append(self.get_i_state(i=i))
            pos_array = self.get_airsim_current_total_pos_array(state_list)
            self.pos = pos_array
            for i in range(self.v_num):  # 更新每个 *无人机* 的位置信息
                self.vehicles[i].get_pos(pos_array[:, self.vehicles[i].ID])
        else:
            print("无法设定无人机群与每个无人机的位置，因为没链接到airsim！")

    def count_uav(self):
        """
        打印本群体中有多少无人机
        :return: 无
        """
        print('共有', len(self.vehicles), '个无人机:', end=' ')
        for v in self.vehicles:
            print(str(v.ID) + '号', end=' ')
        print('\n')

    def show_position(self):
        """
        打印本群体中所有无人机的位置信息
        :return: 无
        """
        for v in self.vehicles:
            print(str(v.ID) + '号', v.pos, '\n')

    def draw_distance(self, total_time):
        """
        测试控制律consensus-based中绘制与0号无人机距离的函数
        :param total_time:  总迭代次数
        :return:            无
        """
        t = [i for i in range(total_time)]  # 用于画图作为时间轴x轴
        for y in self.distance:
            plt.plot(t, y)
        legend_list = ['uav' + str(i) for i in range(self.v_num)]
        plt.legend(legend_list)
        plt.ylabel('distance between UAV0 and others')
        plt.show()

    # ==================================================================================================================
    # ==================================================================================================================
    def iterate(self, iterate_func=None, total_time=50, dt=0.1):
        if self.is_connected():
            self.unlock_all()
            self.takeoff_all()
            self.hangover(2)                    # 前期准备

            # 数据记录+++++++++++++++++++++++++++++++++++++++++++++++++
            # names = ['x', 'y', 'z',
            #          'vx', 'vy', 'vz',
            #          'ax', 'ay', 'az',
            #          'pitch', 'roll', 'yaw',
            #          'pitch_a', 'roll_a', 'yaw_a']
            #
            # single_data_list = [[] for i in range(15)]  # 单个无人机的数据列表，其中有十五个属性
            # data_list = [single_data_list for i in range(self.v_num)]  # i个无人机的数据列表
            # 数据记录+++++++++++++++++++++++++++++++++++++++++++++++++

            begin_time = time.time()

            for step in range(total_time):      # 开始迭代
                self.airsim_update_vehicles()   # 从虚拟环境中获取参数并讲参数值都付给所有无人机

                if iterate_func is None:        # 使用控制律计算应有的速度
                    next_directions = self.test_vicsek()
                    self.update_vehicles_direction(next_directions)

                else:
                    next_directions = iterate_func(step=step, vehicles=self.vehicles)
                    # if isinstance(next_directions, np.ndarray):
                    #     next_directions = np.nan_to_num(next_directions)  # 将nan值替换为0
                    self.update_vehicles_direction(next_directions)

                for i in range(self.v_num):
                    name = "UAV" + str(self.vehicles[i].ID + 1)
                    if i != self.v_num - 1:
                        self.client.moveByVelocityAsync(vx=self.vehicles[i].direction[0][0],
                                                        vy=self.vehicles[i].direction[1][0],
                                                        vz=-0.1,
                                                        duration=dt,
                                                        vehicle_name=name)
                    else:
                        self.client.moveByVelocityAsync(vx=self.vehicles[i].direction[0][0],
                                                        vy=self.vehicles[i].direction[1][0],
                                                        vz=-0.1,
                                                        duration=dt,
                                                        vehicle_name=name).join()

                    _, i_data_j = self.get_i_all(i)

                    d2c = Data2csv()
                    d2c.write(i, i_data_j)

            #         for j in range(15):
            #             data_list[i][j].append(i_data_j[j])  # 第i个无人机的第j个属性
            #
            # print(len(data_list), len(data_list[0]), len(data_list[0][0]))
            # for i in range(self.v_num):  # 每个无人机逐个书写
            #     i_dict = {}
            #     for j in range(15):
            #         i_dict[names[j]] = data_list[i][j]
            #     i_df = pd.DataFrame(i_dict)
            #     i_df.to_csv('UAV' + str(i) + '.csv')

            time.sleep(dt)

            end_time = time.time()
            run_time = end_time - begin_time
            print('该程序运行时间：', run_time)

            self.land_all()
            self.lock_all()
        else:
            for step in range(total_time):
                iterate_func(step)
            self.draw_distance(total_time)
            print('无法迭代，因为没有连接！')
    # ==================================================================================================================
    # ==================================================================================================================

    def airsim_update_vehicles(self):
        """
        从airsim中获取当前无人机们的所有信息，并全部赋值给所有无人机
        :return: 无
        """
        for i in range(self.v_num):
            state = self.get_i_state(self.vehicles[i].ID)
            self.vehicles[i].update_state(state)

    def unlock_all(self):
        """
        获取无人机群的控制权限并解锁无人机群
        :return: 无test
        """
        if self.is_connected():
            for i in range(self.v_num):  # 使所有无人机获取控制权并解锁
                name = "UAV" + str(self.vehicles[i].ID + 1)
                self.client.enableApiControl(True, name)  # 获取控制权
                self.client.armDisarm(True, name)  # 解锁（螺旋桨开始转动）

    def takeoff_all(self):  # 使所有无人机起飞
        """
        使无人机群起飞
        :return: 无
        """
        if self.is_connected():
            for i in range(self.v_num):
                name = "UAV" + str(self.vehicles[i].ID + 1)
                if i != self.v_num - 1:  # 起飞
                    self.client.takeoffAsync(vehicle_name=name)
                else:
                    self.client.takeoffAsync(vehicle_name=name).join()

    def land_all(self):  # 使所有无人机降落
        """
        使无人机群降落
        :return: 无
        """
        if self.is_connected():
            for i in range(self.v_num):
                name = "UAV" + str(self.vehicles[i].ID + 1)
                if i != self.v_num - 1:  # 降落
                    self.client.landAsync(vehicle_name=name)
                else:
                    self.client.landAsync(vehicle_name=name).join()

    def lock_all(self):  # 使所有无人机上锁
        """
        给无人机群上锁（停止旋翼转动）并归还控制权
        :return: 无
        """
        if self.is_connected():
            for i in range(self.v_num):
                name = "UAV" + str(self.vehicles[i].ID + 1)
                self.client.armDisarm(False, vehicle_name=name)  # 上锁
                self.client.enableApiControl(False, vehicle_name=name)  # 释放控制权

    def hangover(self, t=2):
        """
        悬停无人机一段时间
        :param t:
        :return:
        """
        for i in range(self.v_num):
            name = "UAV" + str(self.vehicles[i].ID + 1)
            if i != self.v_num:
                self.client.hoverAsync(vehicle_name=name)  # 第四阶段：悬停6秒钟
            else:
                self.client.hoverAsync(vehicle_name=name).join()  # 第四阶段：悬停6秒钟
        time.sleep(t)

    def connect_to_airsim(self):
        """
        连接airsim，获取client句柄
        :return: 无
        """
        self.client = airsim.MultirotorClient()

    def is_connected(self):
        """
        判断本无人机群是否接入airsim
        :return: True表示已接入，False表示没接入
        """
        if self.client is not None:
            return True
        else:
            return False

    def get_i_state(self, i):
        """
        获取编号为i的无人机在虚幻引擎中的信息
        :param i:   需要获取信息的无人机的编号
        :return:    无人机位置与速度信息的字典
        """
        fly_state = self.client.simGetGroundTruthKinematics(vehicle_name="UAV" + str(i + 1))
        x = fly_state.position.x_val  # 全局坐标系下，x轴方向的坐标
        y = fly_state.position.y_val  # 全局坐标系下，y轴方向的坐标
        z = fly_state.position.z_val  # 全局坐标系下，z轴方向的坐标

        # 无人机全局速度真值
        vx = fly_state.linear_velocity.x_val  # 无人机 x 轴方向 (正北) 的速度大小真值
        vy = fly_state.linear_velocity.y_val  # 无人机 y 轴方向 (正东) 的速度大小真值
        vz = fly_state.linear_velocity.z_val  # 无人机 z 轴方向 (垂直向下) 的速度大小真值
        # 无人机全局加速度真值

        return {"x": x, "y": y, "z": z, "vx": vx, "vy": vy, "vz": vz}

    def test_takeoff_land(self):
        """
        起飞降落测试
        :return: 无
        """
        if self.is_connected():
            self.unlock_all()
            self.takeoff_all()
            self.land_all()
            self.lock_all()

    def test_vicsek(self):
        """
        内置测试用控制算法
        :return:
        """
        # 控制律部分++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sum_sin = 0
        sum_cos = 0
        for v in self.vehicles:  # 类外编写函数时，无人机数量是使用者知道的
            angle = math.atan2(v.vel[1][0], v.vel[0][0])
            sum_sin += math.sin(angle)
            sum_cos += math.cos(angle)

        # 维度只有2时
        next_angle = math.atan(sum_sin / sum_cos)
        next_direction = np.array([[math.cos(next_angle)], [math.sin(next_angle)]])

        noise = np.random.randn(self.dimension)
        noise = np.array([[noise[i]] for i in range(self.dimension)])
        noise = noise / np.linalg.norm(noise)

        next_direction = noise + next_direction

        next_directions = []
        for i in range(self.v_num):
            next_directions.append(next_direction)

        # 控制律部分++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 只要选择将控制律想要控制的参数更新给无人机即可
        return next_directions

    def update_vehicles_direction(self, next_directions):
        """
        更新每个无人机的方向参数
        :param next_directions:下一刻无人机的方向信息
        :return:
        """
        if isinstance(next_directions, list):
            for i in range(self.v_num):
                self.vehicles[i].get_dir(next_directions[i])

        elif isinstance(next_directions, np.ndarray):
            for i in range(self.v_num):
                self.vehicles[i].get_dir(next_directions[:, i])

    def test_speed(self):
        """
        起飞、悬停、速度控制前进一秒
        :return:
        """
        if self.is_connected():
            self.unlock_all()
            self.takeoff_all()

            self.hangover(2)

            for i in range(self.v_num):
                name = "UAV" + str(self.vehicles[i].ID + 1)
                if i != self.v_num - 1:
                    self.client.moveByVelocityAsync(vx=1, vy=1, vz=0, duration=2, vehicle_name=name)
                else:
                    self.client.moveByVelocityAsync(vx=1, vy=1, vz=0, duration=2, vehicle_name=name).join()

            self.land_all()
            self.lock_all()

    def test_line(self):
        """
        位置控制模式 直线飞行
        :return: 无
        """
        if self.is_connected():
            self.unlock_all()
            self.takeoff_all()

            for i in range(self.v_num):
                name = "UAV" + str(self.vehicles[i].ID + 1)
                if i == self.v_num - 1:
                    self.client.moveToZAsync(-2, 1, vehicle_name=name).join()  # 上升到 高度
                else:
                    self.client.moveToZAsync(-2, 1, vehicle_name=name).join()  # 上升到 高度

            for i in range(self.v_num):
                name = "UAV" + str(self.vehicles[i].ID + 1)
                if i == self.v_num - 1:
                    self.client.moveToPositionAsync(5, 0, -2, 1, vehicle_name=name).join()  # 飞到（5,0）点坐标
                else:
                    self.client.moveToPositionAsync(5, 0, -2, 1, vehicle_name=name)  # 飞到（5,0）点坐标
                    state = self.get_i_state(i)
                    print('无人机', i, '的位置信息为：', state['x'], state['y'], state['z'])

            self.land_all()
            self.lock_all()

    def return_vehicles(self):
        return self.vehicles

    def collect_i(self, i, d):
        names, data = self.get_i_all(i)
        for j in range(len(names)):
            d[names[i]].append(data[j])

        return d

    def get_i_all(self, i):
        fly_state = self.client.simGetGroundTruthKinematics(vehicle_name="UAV" + str(i + 1))
        x = fly_state.position.x_val  # 全局坐标系下，x轴方向的坐标
        y = fly_state.position.y_val  # 全局坐标系下，y轴方向的坐标
        z = fly_state.position.z_val  # 全局坐标系下，z轴方向的坐标

        # 无人机全局速度真值
        vx = fly_state.linear_velocity.x_val  # 无人机 x 轴方向 (正北) 的速度大小真值
        vy = fly_state.linear_velocity.y_val  # 无人机 y 轴方向 (正东) 的速度大小真值
        vz = fly_state.linear_velocity.z_val  # 无人机 z 轴方向 (垂直向下) 的速度大小真值

        ax = fly_state.linear_acceleration.x_val  # 无人机 x 轴方向 (正北) 的加速度大小真值
        ay = fly_state.linear_acceleration.y_val  # 无人机 y 轴方向 (正东) 的加速度大小真值
        az = fly_state.linear_acceleration.z_val  # 无人机 z 轴方向 (垂直向下) 的加速度大小真值

        # 机体角速率
        pitch = fly_state.angular_velocity.x_val        # 机体俯仰角速率
        roll = fly_state.angular_velocity.y_val         # 机体滚转角速率
        yaw = fly_state.angular_velocity.z_val          # 机体偏航角速率

        # 机体角加速度
        pitch_a = fly_state.angular_acceleration.x_val    # 机体俯仰角加速度
        roll_a = fly_state.angular_acceleration.y_val     # 机体滚转角加速度
        yaw_a = fly_state.angular_acceleration.z_val      # 机体偏航角加速度

        names = ['x', 'y', 'z',
                 'vx', 'vy', 'vz',
                 'ax', 'ay', 'az',
                 'pitch', 'roll', 'yaw',
                 'pitch_a', 'roll_a', 'yaw_a']

        data = [x, y, z, vx, vy, vz, ax, ay, az, pitch, roll, yaw, pitch_a, roll_a, yaw_a]

        return names, data

