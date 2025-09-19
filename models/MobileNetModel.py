from models.MobileNetBlocks import MobileNetBlock
import torch
import torch.nn as nn
from models.CNNLinearBlocks import CNNBlock, LinearBlock


class MobileNetModel(nn.Module):
    def __init__(self):
        super(MobileNetModel, self).__init__()
        block_type = 'MobileNetV1'
        # CNN Block 配置，使用1D卷积来处理时间序列数据
        BN1 = {'use_BN': False}
        Activation1 = {'activation_type': 'relu'}
        Pooling1 = {'pooling_type': 'max2d', 'pooling_kernel': (2, 1), 'pooling_stride': (2, 1)}
        Dropout1 = {'drop_rate': 0.2}

        BN2 = {'use_BN': False}
        Activation2 = {'activation_type': 'relu'}
        Pooling2 = {'pooling_type': 'max2d', 'pooling_kernel': (2, 2), 'pooling_stride': (2, 2)}
        Dropout2 = {'drop_rate': 0.2}

        # 定义卷积层，使用1D卷积
        Conv1 = {'in_channels': 1, 'out_channels': 32, 'kernel_size': (3,3), 'stride': (1,1), 'dilation': (1, 1), 'padding': 'same'}
        Conv2 = {'in_channels': 32, 'out_channels': 64, 'kernel_size': (3,3), 'stride': (1,1), 'dilation':  (1,1), 'padding': 'same'}
        self.cnn = nn.Sequential(
            MobileNetBlock( mobilenet_dict = Conv1, bn_dict=BN1, pooling_dict=Pooling1, dropout_dict=Dropout1, block_type=block_type),
            MobileNetBlock(mobilenet_dict=Conv2, bn_dict=BN2, pooling_dict=Pooling2, dropout_dict=Dropout2,
                       block_type=block_type),
        )
        ## 32*128*144
        # 共享特征提取后的线性层部分
        linear1_in_dim = 64 * 144  # 计算卷积层输出后的维度
        Linear1 = {'in_dim': linear1_in_dim, 'out_dim': 128}
        BN1_linear = {'use_BN': False}
        Activation1_linear = {'activation_type': 'relu'}
        Dropout1_linear = {'use_dropout': False, 'drop_rate': 0.25}

        # 分类任务的输出：4类
        Linear2_class = {'in_dim': 128, 'out_dim': 4}  # 4类分类
        BN2_linear_class = {'use_BN': False}
        Activation2_linear_class = {'activation_type': 'None'}
        Dropout2_linear_class = {'use_dropout': False, 'drop_rate': 0.25}

        # 回归任务的输出：预测16个数据点
        Linear2_regression = {'in_dim': 128, 'out_dim': 16*4}  # 回归预测未来16个数据
        BN2_linear_regression = {'use_BN': False}
        Activation2_linear_regression = {'activation_type': 'None'}
        Dropout2_linear_regression = {'use_dropout': False, 'drop_rate': 0.25}

        # 定义线性层
        self.classification_branch = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2_class, bn_dict=BN2_linear_class, activation_dict=Activation2_linear_class,
                        dropout_dict=Dropout2_linear_class),
        )

        self.regression_branch = nn.Sequential(
            LinearBlock(linear_dict=Linear1, bn_dict=BN1_linear, activation_dict=Activation1_linear,
                        dropout_dict=Dropout1_linear),
            LinearBlock(linear_dict=Linear2_regression, bn_dict=BN2_linear_regression, activation_dict=Activation2_linear_regression,
                        dropout_dict=Dropout2_linear_regression),
        )
    def forward(self, data):
        # data: [batch_size, 32, 31]
        m1_out = self.cnn(data)  # [batch_size, 128, length]
        # print(cnn_out.shape)
        m1_out = m1_out.view(m1_out.size(0), -1)  # Flatten for linear layers
        # print(cnn_out.shape)

        # 回归任务的输出
        reg_out = self.regression_branch(m1_out)  # 输出形状为 [batch_size, 16 * 4]
        reg_out = reg_out.view(reg_out.size(0), 16, 4)  # 重新调整形状为 [batch_size, 16, 4]
        # 分类任务的输出
        # print(reg_out.shape)
        class_out = self.classification_branch(m1_out)  # [batch_size, 4]
        # print(class_out.shape)
        return reg_out, class_out


    def get_model_name(self):
        return 'MobileNetModel'

#%%
import os