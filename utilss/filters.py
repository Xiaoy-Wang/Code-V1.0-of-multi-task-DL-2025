## EIA 算法 Empirical Iterative Algorithm EMD轻量版
## 文献1：MyoNet: A Transfer-Learning-Based LRCN for Lower Limb Movement Recognition and Knee Joint Angle Prediction for Remote Monitoring of Rehabilitation Progress From sEMG
## 文献2：A Data Driven Empirical Iterative Algorithm for GSR Signal Pre-Processing  https://ieeexplore.ieee.org/document/8553191/references#references
## 文献3：Midpoint-based empirical decomposition for nonlinear trend estimation https://ieeexplore.ieee.org/document/5335028
from scipy.signal import argrelextrema #获取信号序列的极值点
import scipy.interpolate as spi #进行样条差值
import numpy as np
from filterpy.kalman import KalmanFilter
def kalman_filter(data):
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([data[0]])  # 初始状态设为数据的第一个值
    kf.P *= 10000.  # 初始不确定性
    kf.F = np.array([[1]])  # 状态转移矩阵
    kf.H = np.array([[1]])  # 测量矩阵
    kf.R = 10  # 测量噪声
    kf.Q = 0.05  # 过程噪声

    filtered_data = []
    for value in data:
        kf.predict()
        kf.update(value)
        filtered_data.append(kf.x[0])  # 获取滤波后的数据
    return np.array(filtered_data)

def moving_average_filter(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re

class EIAFilter():
    def __init__(self,data,version='V2'):
        self.data= data
        self.version=version
        if self.data.ndim==1:
            self.num=1
        else:
            self.num=self.data.shape[0]

    # @staticmethod 不用实例化直接调用子函数
    def hasPeaks(self,data):
        max_peaks = list(argrelextrema(data, np.greater)[0])
        min_peaks = list(argrelextrema(data, np.less)[0])
        if len(max_peaks)>3 and len(min_peaks)>3:
            return True
        else:
            return False

    def EIAV1(self,data):
        rawdata=data
        smoothdata=[]
        residualdata =[]
        for i in range(self.num):
            if rawdata.ndim==1:
               rawdata=rawdata[np.newaxis,:]
            signal=rawdata[i,:]
            smoothsignal=self.EIAV1Single(signal)
            residualsignal=signal-smoothsignal
            smoothdata.append(smoothsignal)
            residualdata.append(residualsignal)
        smoothdata=np.array(smoothdata)
        residualdata=np.array(residualdata)
        return  smoothdata,residualdata

    def EIAV2(self,data):
        rawdata=data
        smoothdata=[]
        residualdata =[]
        for i in range(self.num):
            if rawdata.ndim==1:
               rawdata=rawdata[np.newaxis,:]
            signal=rawdata[i,:]
            smoothsignal=self.EIAV2Single(signal)
            residualsignal=signal-smoothsignal
            smoothdata.append(smoothsignal)
            residualdata.append(residualsignal)
        smoothdata=np.array(smoothdata)
        residualdata=np.array(residualdata)
        return smoothdata,residualdata

    def EIAV1Single(self,signal):
        reconstructedSignal = []
        if self.hasPeaks(signal):
            while self.hasPeaks(signal):
                # print('Iuput signal has peaks!')
                ## step1 计算局部极大值点,局部极小值点
                index = list(range(len(signal)))
                max_peaks = list(argrelextrema(signal, np.greater)[0])
                min_peaks = list(argrelextrema(signal, np.less)[0])
                max_peaks_data=signal[max_peaks]
                min_peaks_data=signal[min_peaks]
                ## step2 计算中点
                # print(len(max_peaks_data),len(min_peaks_data))
                length=min(len(max_peaks_data),len(min_peaks_data))
                middle_points= [(max_peaks[k]+min_peaks[k])/2 for k in range(length)]
                middle_points_data=[(max_peaks_data[k]+min_peaks_data[k])/2 for k in range(length)]
                ## step3 对中点进行三次样条插值拟合曲线
                # print(middle_points)
                ipo3_middle_points = spi.splrep(middle_points, middle_points_data,k=3) #样本点导入，生成参数
                iy3__middle_data = spi.splev(index, ipo3_middle_points) #根据观测点和样条参数，生成插值
                signal = iy3__middle_data
            reconstructedSignal =np.array(signal)
        else:
            reconstructedSignal =np.array(signal)
        return reconstructedSignal

    def EIAV2Single(self,signal):
        reconstructedSignal = []
        if self.hasPeaks(signal):
            while self.hasPeaks(signal):
                signal = self.sifting(signal)
            reconstructedSignal = signal
        else:
            reconstructedSignal = signal
        return reconstructedSignal

    def sifting(self,data):
        index = list(range(len(data)))
        max_peaks = list(argrelextrema(data, np.greater)[0])
        min_peaks = list(argrelextrema(data, np.less)[0])
        ipo3_max = spi.splrep(max_peaks, data[max_peaks],k=3) #样本点导入，生成参数
        iy3_max = spi.splev(index, ipo3_max) #根据观测点和样条参数，生成插值
        ipo3_min = spi.splrep(min_peaks, data[min_peaks],k=3) #样本点导入，生成参数
        iy3_min = spi.splev(index, ipo3_min) #根据观测点和样条参数，生成插值
        iy3_mean = (iy3_max+iy3_min)/2
        return iy3_mean

    def build(self):
        if self.version=='V1':
            smoothdata,residualdata=self.EIAV1(self.data)
        elif self.version=='V2':
            smoothdata,residualdata=self.EIAV2(self.data)
        return smoothdata
