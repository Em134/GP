import numpy as np
import math


class Algorithm(object):
    def __init__(self):
        """
        继承这个类，编写控制算法。
        本类中内置三个算法供参考。

        格式举例：
        def test_line(self, vehicles, step=1):

            # 算法的计算方式从这里开始
            d = []
            self.p = 1
            for i in range(len(vehicles)):
            d.append(np.array([[1.0], [0.0]]))
            # 算法的计算方式到这里结束

            # 返回所有无人机所得到的调整结果，例如速度等
            return d
        """
        self.p = 1
        self.dimension = 2
        self.A = np.array([[0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 0],
                          [0, 1, 0, 0, 1, 1],
                          [0, 1, 0, 0, 1, 0],
                          [0, 1, 1, 1, 0, 1],
                          [0, 0, 1, 0, 1, 0]
                           ])
        pass

    def vicsek(self, vehicles, step=1):
        sum_sin = 0
        sum_cos = 0
        for v in vehicles:  # 类外编写函数时，无人机数量是使用者知道的
            angle = math.atan2(v.vel[1][0], v.vel[0][0])
            sum_sin += math.sin(angle)
            sum_cos += math.cos(angle)

        # 维度只有2时
        next_angle = math.atan(sum_sin / sum_cos)
        next_direction = np.array([[math.cos(next_angle)], [math.sin(next_angle)]])
        next_directions = []

        noise = np.random.randn(self.dimension)
        noise = np.array([[noise[i]] for i in range(self.dimension)])
        noise = noise / np.linalg.norm(noise)

        next_direction = next_direction + noise

        for i in range(len(vehicles)):
            next_directions.append(next_direction)

        return next_directions

    def vision_based(self, vehicles, step=1):
        # 算法内容=======================================================================================================
        # 参数设置
        origin_x = [0, 2, 4, 0, 2, 4, 0, 2, 4]  # 无人机的期望阵型
        origin_y = [0, 0, 0, -3, -2, -3, 3, 2, 3]

        v_num = len(vehicles)
        self.p = 2  # 没用的变量，只是为了不是静态方法
        v_max = 2   # 无人机最大飞行速度
        r_max = 20  # 邻居选择的半径
        k_sep = 7   # 避碰系数    7
        k_coh = 1   # 群集系数    1
        k_mig = 1   # 整体迁移系数 1
        pos_mig = np.array([[25], [0]])  # 目标位置
        v_cmd = np.zeros([2, len(vehicles)])

        for i in range(v_num):  # 计算每个无人机的速度指令
            # print(np.array([[origin_x[i]], origin_y[i]]))
            pos_i = vehicles[i].pos + np.array([[origin_x[i]], [origin_y[i]]])
            r_mig = pos_mig - pos_i
            v_mig = k_mig * r_mig / np.linalg.norm(r_mig)
            v_sep = np.zeros([2, 1])
            v_coh = np.zeros([2, 1])
            N_i = 0
            for j in range(v_num):
                if j != i:
                    N_i += 1
                    pos_j = vehicles[j].pos + np.array([[origin_x[j]], [origin_y[j]]])
                    if np.linalg.norm(pos_j - pos_i) < r_max:
                        r_ij = pos_j - pos_i
                        v_sep += -k_sep * r_ij / np.linalg.norm(r_ij)
                        v_coh += k_coh * r_ij
            v_sep = v_sep / N_i
            v_coh = v_coh / N_i
            v_cmd[:, i:i + 1] = v_sep + v_coh + v_mig
        # 算法内容=======================================================================================================

        # 返回结果
        # print(v_cmd)
        return v_cmd

    def vision_based_cb(self, vehicles, step=1, alpha=0.1, u_star=np.array([[1.0], [0.0]])):
        # 算法内容=======================================================================================================
        # 参数设置
        origin_x = [0, 2, 4, 0, 2, 4, 0, 2, 4]  # 无人机的期望阵型
        origin_y = [0, 0, 0, -3, -2, -3, 3, 2, 3]

        v_num = len(vehicles)
        self.p = 2  # 没用的变量，只是为了不是静态方法
        v_max = 2   # 无人机最大飞行速度
        r_max = 20  # 邻居选择的半径
        k_sep = 7   # 避碰系数    7
        k_coh = 1   # 群集系数    1
        k_mig = 1   # 整体迁移系数 1
        pos_mig = np.array([[25], [0]])  # 目标位置
        v_cmd = np.zeros([2, len(vehicles)])

        for i in range(v_num):  # 计算每个无人机的速度指令
            # print(np.array([[origin_x[i]], origin_y[i]]))
            pos_i = vehicles[i].pos + np.array([[origin_x[i]], [origin_y[i]]])
            r_mig = pos_mig - pos_i
            v_mig = k_mig * r_mig / np.linalg.norm(r_mig)

            s = np.zeros((self.dimension, 1))  # 用于计算求和部分
            for k in range(v_num):
                pos_k = vehicles[k].pos + np.array([[origin_x[k]], [origin_y[k]]])
                s = s + pos_k - pos_i
            v_mig = u_star + alpha * s
            v_mig = v_mig / np.linalg.norm(v_mig)

            v_sep = np.zeros([2, 1])
            v_coh = np.zeros([2, 1])
            N_i = 0
            for j in range(v_num):
                if j != i:
                    N_i += 1
                    pos_j = vehicles[j].pos + np.array([[origin_x[j]], [origin_y[j]]])
                    if np.linalg.norm(pos_j - pos_i) < r_max:
                        r_ij = pos_j - pos_i
                        v_sep += -k_sep * r_ij / np.linalg.norm(r_ij)
                        v_coh += k_coh * r_ij
            v_sep = v_sep / N_i
            v_coh = v_coh / N_i
            v_cmd[:, i:i + 1] = v_sep + v_coh + v_mig
        # 算法内容=======================================================================================================

        # 返回结果
        # print(v_cmd)
        return v_cmd

    def test_line(self, vehicles, step=1):
        d = []
        self.p = 1
        for i in range(len(vehicles)):
            d.append(np.array([[1.0], [0.0]]))

        return d

    def displacement_based(self, vehicles, step=1, v_star=np.array([[1], [0]])):
        v_num = len(vehicles)
        a_cmd = np.zeros([2, v_num])

        # origin_x = [0, 2, 4, 0, 2, 4, 0, 2, 4]    # 无人机的期望阵型/相对位置
        # origin_y = [0, 0, 0, -3, -2, -3, 3, 2, 3]
        origin_x = [0, 10, 20, 10, 20, 20]
        origin_y = [20, 20, 20, 10, 10, 0]
        delta = np.array([origin_x, origin_y])

        dt = 0.1
        alpha = 0.5
        beta = 1.7
        g = np.array([[0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [0, 1, 0, 1, 1, 0],
                      [0, 0, 1, 0, 1, 1],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0]
                      ])
        # g = np.array([[1 for _ in range(v_num)] for _ in range(v_num)])

        for i in range(v_num):  # 计算每个无人机的速度指令
            summary = np.zeros(shape=(self.dimension, 1))
            for j in range(v_num):
                noise_v = np.random.randn(self.dimension)
                noise_v = np.array([[noise_v[i]] for i in range(noise_v.shape[0])])
                noise_p = np.random.randn(self.dimension)
                noise_p = np.array([[noise_p[i]] for i in range(noise_p.shape[0])])

                delta_v = vehicles[i].direction - vehicles[j].direction + noise_v
                delta_p = (vehicles[i].pos - delta[:, i: i + 1]) - (vehicles[j].pos - delta[:, j: j + 1]) + noise_p

                summary = summary + g[i, j] * (beta * delta_v + delta_p)

            u_i = -alpha * (vehicles[i].direction - v_star) - summary

            a_cmd[:, i:i + 1] = vehicles[i].direction + u_i * dt

        return a_cmd

    def vision_based_small(self, vehicles, step=1):
        # 算法内容=======================================================================================================
        # 参数设置
        origin_x = [0, 2, 4, 0, 2, 4, 0, 2, 4]  # 无人机的期望阵型
        origin_y = [0, 0, 0, -3, -2, -3, 3, 2, 3]

        v_num = len(vehicles)
        self.p = 2  # 没用的变量，只是为了不是静态方法
        v_max = 2   # 无人机最大飞行速度
        r_max = 20  # 邻居选择的半径
        k_sep = 20   # 避碰系数    7
        k_coh = 1   # 群集系数    1
        k_mig = 1   # 整体迁移系数 1
        pos_mig = np.array([[25], [0]])  # 目标位置
        v_cmd = np.zeros([2, len(vehicles)])

        for i in range(v_num):  # 计算每个无人机的速度指令
            # print(np.array([[origin_x[i]], origin_y[i]]))
            pos_i = vehicles[i].pos + np.array([[origin_x[i]], [origin_y[i]]])
            r_mig = pos_mig - pos_i
            v_mig = k_mig * r_mig / np.linalg.norm(r_mig)
            v_sep = np.zeros([2, 1])
            v_coh = np.zeros([2, 1])
            N_i = 0
            for j in range(v_num):
                if j != i:
                    N_i += 1
                    pos_j = vehicles[j].pos + np.array([[origin_x[j]], [origin_y[j]]])
                    if np.linalg.norm(pos_j - pos_i) < r_max:
                        r_ij = pos_j - pos_i
                        v_sep += -k_sep * r_ij / np.linalg.norm(r_ij)
                        v_coh += k_coh * r_ij
            v_sep = v_sep / N_i
            v_coh = v_coh / N_i
            v_cmd[:, i:i + 1] = v_sep + v_coh + v_mig
        # 算法内容=======================================================================================================

        # 返回结果
        # print(v_cmd)
        return v_cmd