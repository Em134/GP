import csv
import pandas as pd
import os


class Data2csv(object):
    def __init__(self):
        self.path = './data/uav'
        self.name = ['x', 'y', 'z',
                     'vx', 'vy', 'vz',
                     'ax', 'ay', 'az',
                     'pitch', 'roll', 'yaw',
                     'pitch_a', 'roll_a', 'yaw_a']

    def create_csv(self, i):
        path = self.path + str(i) + '.csv'
        data_df = pd.DataFrame(columns=self.name)
        data_df.to_csv(path)

    def write(self, i, data):
        path = self.path + str(i) + '.csv'
        if not os.path.exists(path):
            self.create_csv(i)
        df = pd.DataFrame([data])
        df.to_csv(path, sep=',', mode='a', header=False)


def data2csv(i, data):
    # 保存数据到csv文件
    new_item_csv = 'uav' + str(i)
    with open('./data/{}.csv'.format(new_item_csv), 'a+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for item in data:
            writer.writerow([item])


