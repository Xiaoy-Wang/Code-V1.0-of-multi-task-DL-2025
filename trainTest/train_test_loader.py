import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader


# 自定义Dataset类
class MyDataset(Dataset):
    def __init__(self, data, label1, label2):
        super().__init__()
        self.data = data
        self.label1 = label1
        self.label2 = label2

    def __getitem__(self, item):
        data = torch.from_numpy(self.data[item])
        label1 = torch.from_numpy(self.label1[item])
        label2 = torch.from_numpy(self.label2[item]).to(dtype=torch.long)
        return data, label1, label2

    def __len__(self):
        return self.data.shape[0]  # 返回数据集的大小
# 数据划分函数
def split_data(file_name, n_repeats):
    with open(file_name, 'rb') as f:
        emg_sample = np.load(f)['emg_sample']
        imu_sample = np.load(f)['imu_sample']
        angle_sample = np.load(f)['angle_sample']
        angle_labels = np.load(f)['label1']
        phase_labels = np.load(f)['label2']

        # 合并EMG和IMU样本
        input_data = np.concatenate((emg_sample, imu_sample), axis=2)  # 共16+21=37通道

        # 设定测试集比例和重复次数
        test_ratio = 0.2
        valid_ratio = 0.1 / (1 - test_ratio)

        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_ratio, random_state=42)
        # 用于存储训练集和测试集数据
        train_data, train_labels1, train_labels2 = [], [], []
        test_data, test_labels1, test_labels2 = [], [], []

        # 划分数据集
        for train_index, test_index in sss.split(input_data, phase_labels):
            train_input_data = input_data[train_index]
            train_angle_labels = angle_labels[train_index]
            train_phase_labels = phase_labels[train_index]
            test_input_data = input_data[test_index]
            test_angle_labels = angle_labels[test_index]
            test_phase_labels = phase_labels[test_index]

            # 将训练集数据添加到列表中
            train_data.append(train_input_data)
            train_labels1.append(train_angle_labels)
            train_labels2.append(train_phase_labels)

            # 将测试集数据添加到列表中
            test_data.append(test_input_data)
            test_labels1.append(test_angle_labels)
            test_labels2.append(test_phase_labels)

        # 转换为 NumPy 数组
        train_data = np.array(train_data)
        train_labels1 = np.array(train_labels1)
        train_labels2 = np.array(train_labels2)
        test_data = np.array(test_data)
        test_labels1 = np.array(test_labels1)
        test_labels2 = np.array(test_labels2)

        return train_data, train_labels1, train_labels2, test_data, test_labels1, test_labels2

# 数据划分函数
def split_data_train_test_valild(file_name, n_repeats):
    with open(file_name, 'rb') as f:
        emg_sample = np.load(f)['emg_sample']
        imu_sample = np.load(f)['imu_sample']
        angle_sample = np.load(f)['angle_sample']
        angle_labels = np.load(f)['label1']
        phase_labels = np.load(f)['label2']

        # 合并EMG和IMU样本
        input_data = np.concatenate((emg_sample, imu_sample), axis=2)  # 共16+21=37通道

        # 设定测试集比例和重复次数
        test_ratio = 0.2
        valid_ratio = 0.1 / (1 - test_ratio)

        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_ratio, random_state=42)
        sssForValid = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=42)
        # 用于存储训练集和测试集数据
        train_data, train_labels1, train_labels2 = [], [], []
        test_data, test_labels1, test_labels2 = [], [], []
        valid_data, valid_labels1, valid_labels2 = [], [], []

        # 划分数据集
        for train_index, test_index in sss.split(input_data, phase_labels):
            trainAndValid_input_data = input_data[train_index]
            trainAndValid_angle_labels = angle_labels[train_index]
            trainAndValid_phase_labels = phase_labels[train_index]
            test_input_data = input_data[test_index]
            test_angle_labels = angle_labels[test_index]
            test_phase_labels = phase_labels[test_index]
            # 将测试集数据添加到列表中
            test_data.append(test_input_data)
            test_labels1.append(test_angle_labels)
            test_labels2.append(test_phase_labels)
            for train_index_, valid_index_ in sssForValid.split(trainAndValid_input_data, trainAndValid_phase_labels):
                train_input_data = trainAndValid_input_data[train_index_]
                train_angle_labels = trainAndValid_angle_labels[train_index_]
                train_phase_labels = trainAndValid_phase_labels[train_index_]
                valid_input_data = trainAndValid_input_data[valid_index_]
                valid_angle_labels = trainAndValid_angle_labels[valid_index_]
                valid_phase_labels = trainAndValid_phase_labels[valid_index_]
                # 将训练集数据添加到列表中
                train_data.append(train_input_data)
                train_labels1.append(train_angle_labels)
                train_labels2.append(train_phase_labels)
                # 将验证集数据添加到列表中
                valid_data.append(valid_input_data)
                valid_labels1.append(valid_angle_labels)
                valid_labels2.append(valid_phase_labels)

        # 转换为 NumPy 数组
        train_data = np.array(train_data)
        train_labels1 = np.array(train_labels1)
        train_labels2 = np.array(train_labels2)
        test_data = np.array(test_data)
        test_labels1 = np.array(test_labels1)
        test_labels2 = np.array(test_labels2)
        valid_data = np.array(valid_data)
        valid_labels1 = np.array(valid_labels1)
        valid_labels2 = np.array(valid_labels2)

        return train_data, train_labels1, train_labels2, test_data, test_labels1, test_labels2,valid_data,valid_labels1,valid_labels2


# DataLoader类，整合数据集加载
class MyDataLoader:
    def __init__(self, file_name, n_repeats, train_batch, test_batch, exp_time):
        # 获取划分后的数据
        # self.train_data, self.train_labels1, self.train_labels2, \
        #     self.test_data, self.test_labels1, self.test_labels2 = split_data(file_name, n_repeats)
        self.train_data, self.train_labels1, self.train_labels2, self.test_data, self.test_labels1, self.test_labels2,\
            self.valid_data, self.valid_labels1, self.valid_labels2 = split_data_train_test_valild(file_name, n_repeats)
        self.exp_time = exp_time
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.valid_batch = test_batch

    def get_loaders(self):
        # 根据exp_time选择对应的数据
        train_set = MyDataset(data=self.train_data[self.exp_time - 1],
                              label1=self.train_labels1[self.exp_time - 1],
                              label2=self.train_labels2[self.exp_time - 1])

        test_set = MyDataset(data=self.test_data[self.exp_time - 1],
                             label1=self.test_labels1[self.exp_time - 1],
                             label2=self.test_labels2[self.exp_time - 1])

        valid_set = MyDataset(data=self.valid_data[self.exp_time - 1],
                             label1=self.valid_labels1[self.exp_time - 1],
                             label2=self.valid_labels2[self.exp_time - 1])

        # 创建DataLoader对象
        train_loader = DataLoader(train_set, batch_size=self.train_batch, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.test_batch, shuffle=False)
        valid_loader = DataLoader(valid_set, batch_size=self.valid_batch, shuffle=False)

        return train_loader, test_loader,valid_loader

