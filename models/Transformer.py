import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.model_name = 'Transformer'

        # 输入维度 37 -> 输出维度 64
        self.embedding = nn.Embedding(37, 64)  # 假设 37 是词汇量大小，64 是嵌入维度

        # 位置编码（假设最大长度为32，嵌入维度为64）
        self.position_embedding = self.getPositionEmbedding(32, 64)  # 假设位置编码为 32 * 64

        ################ TRANSFORMER BLOCK ###################
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,  # 输入特征维度
            nhead=4,  # 多头注意力的头数
            dim_feedforward=128,  # 前馈网络的维度
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=64,  # 解码器输入的维度
            nhead=4,  # 解码器的多头注意力头数
            dim_feedforward=128,  # 前馈网络的维度
            dropout=0.2,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)

        # 分类和回归层
        self.classifier = nn.Linear(64, 4)  # 4分类任务
        self.angle_predictor = nn.Linear(64, 16)  # 16个角度预测

    def getPositionEmbedding(self, length, dim):
        # 创建位置编码
        position = torch.arange(0, length).unsqueeze(1).float()  # (length, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / dim))  # 计算频率
        pe = torch.zeros(length, dim)  # 初始化位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引
        pe = pe.unsqueeze(0)  # (1, length, dim)
        return pe

    def forward(self, x):
        # x 的形状: [batch_size, 32, 37]
        batch_size = x.shape[0]

        # 嵌入：将输入x转化为64维
        embedding = self.embedding(x)  # [batch_size, 32, 64]

        # 加上位置编码
        position_embedding = self.position_embedding[:, :x.shape[1], :]  # 位置编码 [1, 32, 64]
        embedding += position_embedding

        # Encoder: 获取输入的上下文表示
        encoder_out = self.transformer_encoder(embedding)  # [batch_size, 32, 64]

        ############################# 分类任务 ##############################
        # 分类任务：取Encoder输出的[32]维度并进行池化
        encoder_out_for_classification = torch.mean(encoder_out, dim=1)  # [batch_size, 64]

        # 分类输出
        class_output = self.classifier(encoder_out_for_classification)  # [batch_size, 4]

        ############################# 角度预测任务 ##############################
        # Decoder 任务：预测未来16个角度
        # 假设 Decoder 的输入是 encoder_out 的最后一个位置，作为初始解码状态
        decoder_input = torch.zeros(batch_size, 16, 64).to(embedding.device)  # [batch_size, 16, 64] 初始化为零

        # 使用 Encoder 输出作为 Decoder 的 memory
        decoder_out = self.transformer_decoder(decoder_input, encoder_out)  # [batch_size, 16, 64]

        # 从 Decoder 输出中提取出需要的预测值（16个角度）
        angle_output = self.angle_predictor(decoder_out[:, -1, :])  # 使用最后一个位置的 Decoder 输出进行预测 [batch_size, 16]

        return class_output, angle_output

    def get_model_name(self):
        return self.model_name