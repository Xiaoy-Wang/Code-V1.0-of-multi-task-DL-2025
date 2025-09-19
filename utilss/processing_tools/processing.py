import math
import numpy as np
from scipy.signal import butter, lfilter, iirfilter
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
from collections import Counter
from scipy.signal import butter, filtfilt
import pandas as pd
from scipy import stats
import math
from utilss.common_utils import all_elements_equal_to_str, is_nan_in_df_rows, analyze_list, get_middle_value_in_list

### 通用工具
"""emg滤波器：陷波滤波、带通滤波、低通滤波"""


class emg_filtering():
    def __init__(self, fs, lowcut, highcut, imf_band, imf_freq):
        self.fs = fs
        # butterWorth带通滤波器
        self.lowcut, self.highcut = lowcut, highcut
        # 50 Hz陷波滤波器
        self.imf_band, self.imf_freq = imf_band, imf_freq
        # 低通滤波
        self.cutoff = 20

    def Implement_Notch_Filter(self, data, order=2, filter_type='butter'):
        # Required input defintions are as follows;
        # time:   Time between samples
        # band:   The bandwidth around the centerline freqency that you wish to filter
        # freq:   The centerline frequency to be filtered
        # ripple: The maximum passband ripple that is allowed in db
        # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
        #         IIR filters are best suited for high values of order.  This algorithm
        #         is hard coded to FIR filters
        # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
        # data:         the data to be filtered
        fs = self.fs
        nyq = fs / 2.0
        freq, band = self.imf_freq, self.imf_band
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype=filter_type)
        filtered_data = lfilter(b, a, data)

        return filtered_data

    def butter_bandpass(self, order=6):
        lowcut, highcut, fs = self.lowcut, self.highcut, self.fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data):
        b, a = self.butter_bandpass()
        y = lfilter(b, a, data)

        return y

    def butter_lowpass(self, order=5):
        cutoff, fs = self.cutoff, self.fs
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        return b, a

    def butter_lowpass_filter(self, data):
        b, a = self.butter_lowpass()
        y = lfilter(b, a, data)

        return y


"""多模态多通道数据归一化方法，其中支持归一化方法：'min-max'、'max-abs'、'positive_negative_one；归一化层面：'matrix'、'rows'"""
def data_nomalize(data, normalize_method, normalize_level):
    if normalize_level == 'matrix':
        if normalize_method == 'min-max':
            # 实例化 MinMaxScaler 并设置归一化范围为 [0, 1]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
        elif normalize_method == 'positive_negative_one':
            # 实例化 MinMaxScaler 并设置归一化范围为 [-1, 1]
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 来进行整体归一化
            # print(np.min(scaler.data_min_),np.max(scaler.data_max_))
            normalized_data = ((data - np.min(scaler.data_min_)) / (
                    np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
        elif normalize_method == 'max-abs':
            # 实例化 MaxAbsScaler，并拟合数据以计算每列的最大值和最小值的绝对值
            scaler = MaxAbsScaler()
            scaler.fit(data)
            # 使用 scaler 的 data_max_ 和 data_min_ 将数据整体缩放到 [-1, 1] 范围内
            normalized_data = data / np.max(np.abs(data))
        else:
            print('Error: 未识别的normalize_method！')
    elif normalize_level == 'rows':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        else:
            print('Error: 未识别的normalize_method！')
    else:
        print('Error: 未识别的normalize_level！')

    return normalized_data, scaler

# % step1：求解四块肌肉的激活度 %
def emg_activation(fs, raw,  d_0, lamda1_0, lamda2_0, A_0):
    output_act = []
    e1 = []
    for i in range(16):
        # Raw EMG signal and MVC signal
        EMG_raw = np.abs(raw[:, i])
        # EMG_MVC = np.abs(max(raw[:, i]))

        # High-pass filter with a cutoff frequency of 30Hz
        bb1, ba1 = butter(4, 30 / (fs / 2), btype='high')

        # Preprocess the EMG signal using zero-phase filtering
        pre_emg_raw = filtfilt(bb1, ba1, EMG_raw)
        pre_emg_raw = np.flip(np.flip(pre_emg_raw, axis=0), axis=0)
        pre_emg_raw = filtfilt(bb1, ba1, pre_emg_raw)

        # pre_emg_mvc = filtfilt(bb1, ba1, EMG_MVC)
        # pre_emg_mvc = np.flip(np.flip(pre_emg_mvc, axis=0), axis=0)
        # pre_emg_mvc = filtfilt(bb1, ba1, pre_emg_mvc)

        # Full-wave rectification (take absolute value)
        pre_emg_raw = np.abs(np.flip(np.flip(pre_emg_raw, axis=0), axis=0))
        # pre_emg_mvc = np.abs(np.flip(np.flip(pre_emg_mvc, axis=0), axis=0))

        # Low-pass filter with a cutoff frequency of 8Hz
        bb2, ba2 = butter(4, 8 / (fs / 2), btype='low')

        pre_emg_raw = filtfilt(bb2, ba2, pre_emg_raw)
        pre_emg_raw = np.flip(np.flip(pre_emg_raw, axis=0), axis=0)
        pre_emg_raw = filtfilt(bb2, ba2, pre_emg_raw)

        # pre_emg_mvc = filtfilt(bb2, ba2, pre_emg_mvc)
        # pre_emg_mvc = np.flip(np.flip(pre_emg_mvc, axis=0), axis=0)
        # pre_emg_mvc = filtfilt(bb2, ba2, pre_emg_mvc)

        # Normalize the raw EMG signal using the MVC
        pre_emg_mvc_max = np.max(abs(pre_emg_raw))

        e = pre_emg_raw / pre_emg_mvc_max

        # Neural activation model
        lamda1 = lamda1_0
        lamda2 = lamda2_0
        bata1 = lamda1 + lamda2
        bata2 = lamda1 * lamda2
        arfa = 1 + bata1 + bata2
        d = max(d_0, 3)  # Ensure d is large enough
        u = np.zeros(len(e))  # Ensure u is large enough to store all values
        d = int(d)  # 强制转换为整数
        for j in range(d, len(e)):
            u[j] = arfa * e[j] - bata1 * u[j - 1] - bata2 * u[j - 2]
        # Muscle activation model using a non-linear equation
        A = A_0
        output_act1 = (np.exp(A * u) - 1) / (np.exp(A) - 1)

        e1.append(e)
        output_act.append(output_act1)
    output = np.array(output_act).T
    return output
# def data_nomalize(data, normalize_method, normalize_level):
#     """
#     对数据进行归一化
#     :param data: 原始数据 (numpy 数组)
#     :param normalize_method: 归一化方法 ('min-max', 'positive-negative-one', 'max-abs')
#     :param normalize_level: 归一化级别 ('matrix', 'rows')
#     :return: 归一化后的数据及 scaler
#     """
#     if normalize_level == 'matrix':
#         if normalize_method == 'min-max':
#             # 实例化 MinMaxScaler 并设置归一化范围为 [0, 1]
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             scaler.fit(data)
#             normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
#         elif normalize_method == 'positive-negative-one':
#             # 实例化 MinMaxScaler 并设置归一化范围为 [-1, 1]
#             scaler = MinMaxScaler(feature_range=(-1, 1))
#             scaler.fit(data)
#             normalized_data = ((data - np.min(scaler.data_min_)) / (
#                     np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
#         elif normalize_method == 'max-abs':
#             # 实例化 MaxAbsScaler，并拟合数据
#             scaler = MaxAbsScaler()
#             scaler.fit(data)
#             normalized_data = data / np.max(np.abs(data))
#         else:
#             print('Error: 未识别的normalize_method！')
#             return None, None
#     elif normalize_level == 'rows':
#         if normalize_method == 'min-max':
#             scaler = MinMaxScaler(feature_range=(0, 1))
#             scaler.fit(data)
#             normalized_data = scaler.transform(data)
#         elif normalize_method == 'positive-negative-one':
#             scaler = MinMaxScaler(feature_range=(-1, 1))
#             scaler.fit(data)
#             normalized_data = scaler.transform(data)
#         elif normalize_method == 'max-abs':
#             scaler = MaxAbsScaler()
#             scaler.fit(data)
#             normalized_data = scaler.transform(data)
#         else:
#             print('Error: 未识别的normalize_method！')
#             return None, None
#     else:
#         print('Error: 未识别的normalize_level！')
#         return None, None
#
#     return normalized_data, scaler


def save_scaler(scaler, filename):
    """
    保存 scaler 到文件
    :param scaler: 归一化使用的 scaler 对象
    :param filename: 文件名
    """
    joblib.dump(scaler, filename)


def load_scaler(filename):
    """
    从文件加载 scaler
    :param filename: 文件名
    :return: 加载的 scaler 对象
    """
    return joblib.load(filename)


def inverse_normalize(normalized_data, scaler, normalize_method):
    """
    对归一化后的数据进行反归一化
    :param normalized_data: 归一化后的数据
    :param scaler: 用于归一化的 scaler 对象
    :param normalize_method: 归一化方法 ('min-max', 'positive-negative-one', 'max-abs')
    :return: 反归一化后的数据
    """
    if normalize_method == 'min-max':
        return scaler.inverse_transform(normalized_data)
    elif normalize_method == 'positive-negative-one':
        # 反向操作：恢复到 [0, 1] 后，再调整为 [-1, 1]
        normalized_data = (normalized_data + 1) / 2
        return scaler.inverse_transform(normalized_data)
    elif normalize_method == 'max-abs':
        return normalized_data * np.max(np.abs(scaler.data_max_))
    else:
        print('Error: 未识别的normalize_method！')
        return None

### 对运动模式分类任务

"""基于滑动重叠窗口采样的样本集分割：重叠窗长window、步进长度step"""


def tj_movement_classification_sample_segmentation(emg_data, imu_data, angle_data, label, window, step, angle_data_initial, verbose=True):
    emg_sample, imu_sample, angle_sample, label_encoded = [], [], [], []
    label_encoded_angle = []

    # 假设 emg_data, imu_data, angle_data 和 angle_data_initial 都是 numpy 数组
    emg = np.array(emg_data)
    imu = np.array(imu_data)
    angle = np.array(angle_data)
    angle_initial = np.array(angle_data_initial)

    # 计算总的滑动窗口数目
    length = math.floor((emg.shape[0] - window - step) / step)

    if verbose:
        print(f"Class:, Number of samples: {length}")
    #
    # for j in range(length):
    #     # 提取每个滑动窗口的 EMG 数据
    #     sub_emg = emg[step * j:(window + step * j), :]
    #     sub_imu = imu[step * j:(window + step * j), :]
    #     sub_angle = angle[step * j:(window + step * j), :]
    #     angle_label = angle[window + step * j: (window + step * j + step), :-2]  # angle为归一化后的数据，angle_initial为没有归一化的数据
    #
    #     # 获取当前窗口的标签（从 label 数组中提取相应位置的标签）
    #     window_label = label[step * j: step * j + window]
    #
    #     # 展平 window_label 并转换为 list 以便检查是否所有标签一致
    #     window_label_flat = window_label.ravel()  # 将 window_label 展平为一维数组
    #
    #     # 使用 Counter 计算标签频率
    #     label_counts = Counter(window_label_flat)
    #
    #     # 如果标签一致，则直接保存当前窗口数据
    #     if len(label_counts) == 1:  # 标签一致
    #         label_encoded.append(window_label_flat[0])  # 将该标签添加到 label_encoded 中
    #         emg_sample.append(sub_emg)
    #         imu_sample.append(sub_imu)
    #         angle_sample.append(sub_angle)
    #         label_encoded_angle.append(angle_label)
    #     else:  # 如果标签不一致，选择频率最多的标签
    #         most_common_label, _ = label_counts.most_common(1)[0]  # 获取出现次数最多的标签
    #         label_encoded.append(most_common_label)  # 使用频率最多的标签作为该窗口的标签
    #         emg_sample.append(sub_emg)
    #         imu_sample.append(sub_imu)
    #         angle_sample.append(sub_angle)
    #         label_encoded_angle.append(angle_label)
    for j in range(length):
        # 提取每个滑动窗口的 EMG 数据
        sub_emg = emg[step * j:(window + step * j), :]
        sub_imu = imu[step * j:(window + step * j), :]
        sub_angle = angle[step * j:(window + step * j), :]
        angle_label = angle[window + step * j: (window + step * j + step), :-2]##angle为归一化后的数据,angle_initial为没有归一化的数据

        # 获取当前窗口的标签（从 label 数组中提取相应位置的标签）
        window_label = label[step * j: step * j + window]

        # 展平 window_label 并转换为 list 以便检查是否所有标签一致
        window_label_flat = window_label.ravel()  # 将 window_label 展平为一维数组

        # 如果标签一致，则保存窗口数据，否则跳过
        if len(set(window_label_flat)) == 1:  # 如果窗口中的标签一致
            label_encoded.append(window_label_flat[0])  # 将该标签添加到 label_encoded 中
            emg_sample.append(sub_emg)
            imu_sample.append(sub_imu)
            angle_sample.append(sub_angle)
            label_encoded_angle.append(angle_label)
        else:
            continue  # 如果标签不一致，跳过当前窗口及其相关数据



    # 将样本数据转换为 numpy 数组
    emg_sample = np.array(emg_sample)
    imu_sample = np.array(imu_sample)
    angle_sample = np.array(angle_sample)

    # 处理角度标签，确保维度正确
    label_1 = np.array(label_encoded_angle)  # 角度标签
    label_2 = np.array(label_encoded)  # 运动标签

    return emg_sample, imu_sample, angle_sample, label_1, label_2
#
# def tj_movement_classification_sample_segmentation(emg_data, imu_data, angle_data, label, window, step,angle_data_initial,
#                                                    verbose=True):
#     emg_sample, imu_sample, angle_sample, label_encoded = [], [], [], []
#     label_encoded_angle = []
#     emg = emg_data
#     imu = imu_data
#     angle = angle_data
#     angle_initial = angle_data_initial
#     length = math.floor((emg.shape[0] - window - step) / step)
#
#     if verbose:
#         print("class ", label, " number of samples: ", length)
#
#     for j in range(length):
#         # 提取每个滑动窗口的 EMG 数据
#         sub_emg = emg[step * j:(window + step * j), :]
#         emg_sample.append(sub_emg)
#
#         # 提取每个滑动窗口的 IMU 数据
#         sub_imu = imu[step * j:(window + step * j), :]
#         imu_sample.append(sub_imu)
#
#         # 提取每个滑动窗口的角度数据
#         sub_angle = angle[step * j:(window + step * j), :]
#         angle_sample.append(sub_angle)
#
#         # 为每个窗口分配标签
#         label_encoded.append(label)
#
#         # 提取角度数据的标签（通常是下一时间步的标签，假设是对应的动作类别）
#         angle_label = angle_initial[window + step * j: (window + step * j + step), :]
#         label_encoded_angle.append(angle_label)
#
#     # 将样本数据转换为 numpy 数组
#     emg_sample, imu_sample = np.array(emg_sample), np.array(imu_sample)
#     angle_sample, label_1,label_2= np.array(angle_sample),  np.array(label_encoded_angle),np.array(label_encoded)
#     if verbose:
#         pass
#
#     return emg_sample, imu_sample, angle_sample, label_1 ,label_2