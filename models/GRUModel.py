import torch.nn as nn
from models.CNNLinearBlocks import CNNBlock, LinearBlock

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        # LSTM部分配置
        self.lstm_input_size = 37  # 假设每个时间步的特征维度为 31
        self.lstm_hidden_size = 128  # LSTM的隐藏状态维度
        self.lstm_num_layers = 2  # LSTM的层数
        self.lstm_dropout = 0.2  # LSTM的dropout比率

        # 定义LSTM层，接收输入并输出特征
        self.lstm = nn.GRU(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            batch_first=True,  # [batch_size, seq_len, feature_size]
                            dropout=self.lstm_dropout)

        # 线性层配置
        linear1_in_dim = self.lstm_hidden_size  # LSTM隐藏层大小为 128
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
        Linear2_regression = {'in_dim': 128, 'out_dim': 16 * 4}  # 回归预测未来16个数据
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
            LinearBlock(linear_dict=Linear2_regression, bn_dict=BN2_linear_regression,
                        activation_dict=Activation2_linear_regression,
                        dropout_dict=Dropout2_linear_regression),
        )

    def forward(self, data):
        # data: [batch_size, 32, 31]，32为序列长度，31为每个时间步的特征数量
        lstm_out, (h_n, c_n) = self.lstm(data)  # lstm_out: [batch_size, seq_len, hidden_size]
        # 使用LSTM最后一个时间步的输出作为特征
        lstm_out = lstm_out[:, -1, :]  # 获取最后时间步的隐藏状态作为特征 [batch_size, hidden_size]

        # Flatten for linear layers, should be [batch_size, hidden_size] now
        lstm_out = lstm_out.view(lstm_out.size(0), -1)  # Flattened to [batch_size, 128]
        # print(lstm_out.shape)

        # 回归任务的输出
        reg_out = self.regression_branch(lstm_out)  # 输出形状为 [batch_size, 16 * 4]
        reg_out = reg_out.view(reg_out.size(0), 16, 4)  # 重新调整形状为 [batch_size, 16, 4]

        # 分类任务的输出
        class_out = self.classification_branch(lstm_out)  # [batch_size, 4]

        return reg_out, class_out

    def get_model_name(self):
        return 'GRUModel'
