import torch
import torch.nn as nn
from models.SCAttention import SCAttBlock
import math
# 输入：batch, 1, 9, 128
# 输出：batch, num_classes

num_classes = 4

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, transformer_average, num_classes=4):
        super(TransformerDecoder, self).__init__()
        self.transformer_average = transformer_average
        assert self.transformer_average in ['time_average', 'channel_average']
        self.model_name = 'TransformerDecoder-' + self.transformer_average

        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(37, 64),  # 将输入的37个特征映射到64个特征
        )

        ################ TRANSFORMER BLOCK #############################
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,  # 输入特征维度
            nhead=4,  # 注意力头数
            dim_feedforward=128,  # 前馈网络的维度
            dropout=0.2,
            activation='relu',
            batch_first=True  # 批处理维度放在第一个
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        # 根据transformer_average选择不同的线性输入维度
        linear_in_features = 32 if self.transformer_average == 'time_average' else 64
        self.linear_part = nn.Sequential(
            nn.Linear(in_features=linear_in_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=4),
        )

        self.mR = nn.Sequential(
            nn.Linear(in_features=linear_in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),  # 输出的特征维度为64
        )

        self.init_params()

    def forward(self, x):
        batchSize = x.shape[0]
        # print('x', x.shape)
        # 去掉squeeze(1)，保持输入维度不变
        embedding = self.embedding(x)
        # print('embedding', embedding.shape)

        length = embedding.shape[1]
        dim = embedding.shape[2]
        encoderPositionEmbedding = self.getPositionEmbedding(length, dim).to(device=x.device)

        # 获取Encoder的输出
        encoderInput = embedding + encoderPositionEmbedding
        # print('encoderInput', encoderInput.shape)
        transformer_output = self.transformer_encoder(encoderInput)
        # print('transformer_output', transformer_output.shape)  # 应该是 [32, 32, 64]

        # 计算branch2_out
        if self.transformer_average == 'time_average':
            branch2_out = torch.mean(transformer_output, dim=2)  # [32, 64]
        else:
            branch2_out = torch.mean(transformer_output, dim=1)  # [32, 32]

        # 输出层
        # print('branchout', branch2_out.shape)
        cla = self.linear_part(branch2_out)
        # print('cla', cla.shape)

        # 计算rul
        rul = self.mR(branch2_out)
        # print('rul', rul.shape)

        # 通过reshape调整rul形状为 [batchSize, 16, 4]
        rul = rul.reshape(batchSize, 16, 4)
        # print('rul', rul.shape)

        return rul, cla

    def getPositionEmbedding(self, seqLen, dimension):
        """
        获取位置嵌入
        :param seq_len: 序列长度
        :param d_model: 嵌入维度
        :return: 位置嵌入矩阵
        """
        # 创建一个位置矩阵，形状为 (seq_len, d_model)
        position = torch.arange(seqLen, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, dimension, 2).float() * -(math.log(10000.0) / dimension))  # (d_model/2,)

        # 计算位置嵌入
        pos_embedding = torch.zeros(seqLen, dimension)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pos_embedding[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos

        return pos_embedding

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_model_name(self):
        return self.model_name
