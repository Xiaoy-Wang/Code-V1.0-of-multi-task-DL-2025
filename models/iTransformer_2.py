import torch
from models.SelfAttentions import DotProductAttention,MultiHeadAttention,ScaledDotProductAttention
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
class QKVBlock(nn.Module):
    def __init__(self, inputDim, qkvOutdim, heads, blockOutdim, dropRate):
        """
        :param inputDim:    QKV的输入维度
        :param qkvOutdim:   QkV的输出维度
        :param heads:       QKV的头数
        :param blockOutdim: 注意力机制块最后的输出维度
        """
        super(QKVBlock, self).__init__()
        self.attentionHeads = heads
        self.modelList = nn.ModuleList([])
        for k in range(self.attentionHeads):
            self.modelList.append(QKVAttention(inputDim, qkvOutdim, dropRate))
        self.lastLinear = nn.Sequential(
            nn.Dropout(dropRate),
            nn.Linear(heads * qkvOutdim, blockOutdim),
        )

    def forward(self, x, maskMatrix=None):
        headResultList = []
        for k in range(self.attentionHeads):
            headResultList.append(self.modelList[k](x, maskMatrix))
        result = torch.concat(headResultList, dim=2)
        y = self.lastLinear(result)
        return y

# QKVAttention是单头注意力，多头注意力使用QKVBlock,QKVAttention只处理到获得多头输出
class QKVAttention(nn.Module):
    def __init__(self, inputDim, qkvOutdim, dropRate):
        """
        :param inputDim:  QKV的输入维度
        :param qkvOutdim: QKV的输出维度
        """
        super(QKVAttention, self).__init__()
        self.linearQ = nn.Sequential(
            nn.Dropout(dropRate),
            nn.Linear(inputDim, qkvOutdim),

        )
        self.linearK = nn.Sequential(
            nn.Dropout(dropRate),
            nn.Linear(inputDim, qkvOutdim),

        )
        self.linearV = nn.Sequential(
            nn.Dropout(dropRate),
            nn.Linear(inputDim, qkvOutdim),
        )

    def forward(self, x, maskMatrix=None):
        if maskMatrix is not None:
            Q = self.drop(self.linearQ(x))  # batchSize * length * qkvOutdim
            K = self.drop(self.linearK(x))  # batchSize * length * qkvOutdim
            V = self.drop(self.linearV(x))  # batchSize * length * qkvOutdim
            dimk = torch.tensor(K.shape[2], device=x.device)  # 系数
            K = torch.transpose(K, 1, 2).contiguous()  # batchSize * qkvOutdim * length
            attentionScore = torch.matmul(Q, K) / torch.sqrt(dimk)  # batchSize * length * length
            attentionScore = attentionScore + maskMatrix
            attentionScore = nn.Softmax(dim=2)(attentionScore)
            headResult = torch.matmul(attentionScore, V)
        else:
            # x:batchSize * length * dim
            Q = self.linearQ(x)  # batchSize * length * qkvOutdim
            K = self.linearK(x)  # batchSize * length * qkvOutdim
            V = self.linearV(x)  # batchSize * length * qkvOutdim
            dimk = torch.tensor(K.shape[2], device=x.device)  # 系数
            K = torch.transpose(K, 1, 2).contiguous()  # batchSize * qkvOutdim * length
            attentionScore = torch.matmul(Q, K) / torch.sqrt(dimk)  # batchSize * length * length
            attentionScore = nn.Softmax(dim=2)(attentionScore)
            headResult = torch.matmul(attentionScore, V)
        return headResult

# 一个BertBlock基础单元
class EncoderBlock(nn.Module):
    def __init__(self, inputDim, qkvOutdim, heads, lastLinearMiddle, dropRate):
        super(EncoderBlock, self).__init__()
        self.QKVBlock = QKVBlock(inputDim, qkvOutdim, heads, inputDim,
                                 dropRate)  # 注意因为BERT中有残差连接，所以block的output一定与input相同
        self.layerNorm1 = nn.LayerNorm(inputDim)
        self.layerNorm2 = nn.LayerNorm(inputDim)
        # 创建最后的线性层，2层，relu激活
        self.lastLinear = nn.Sequential(
            nn.Dropout(dropRate),
            nn.Linear(inputDim, lastLinearMiddle),
            nn.ReLU(),
            nn.Dropout(dropRate),
            nn.Linear(lastLinearMiddle, inputDim),
        )

    def forward(self, x, maskMatrix=None):
        QKVBlockResult = self.QKVBlock(x, maskMatrix)  # batchSize * length * inputDim
        y = self.layerNorm1(QKVBlockResult + x)  # batchSize * length * inputDim
        lastLinearOut = self.lastLinear(y)
        result = self.layerNorm2(lastLinearOut + y)
        return result


# 下面这个类是专门复现SPN论文中的BERT部分
# 注意虽然BertPart由多个BertBlock构成，但是除了BertBLock的heads、qkvOutdim、lastLinearMiddle可能不同外，inputDim都必须相同
# 正常情况下，应该把qkvOutdim和heads还有lastLinearMiddle设置成一样的，这里采用这种写法
# 如果要每一层自定义的话，需要修改代码
class EncoderPart(nn.Module):
    def __init__(self, bertLayers, inputDim, qkvOutdim, heads, lastLinearMiddle, dropRate):
        """
        :param bertLayers:          encoder bert 的层数
        :param inputDim:            bert 的输入维度，对应wordembedding的输出维度，且在整个过程中保持不变
        :param qkvOutdim:           qkv注意力得出QKV矩阵时候的张量维度
        :param heads:               qkv有几个头
        :param lastLinearMiddle:    最后一层线性层的中间维度
        """
        super(EncoderPart, self).__init__()
        self.bertlayers = bertLayers
        self.modelList = nn.ModuleList([])
        for k in range(self.bertlayers):
            self.modelList.append(EncoderBlock(inputDim, qkvOutdim, heads, lastLinearMiddle, dropRate))

    def forward(self, sentences, maskMatrix=None):
        # 这里的sentences是embedding之后的sentences
        # sentences : batchSize * length * embeddingDim
        x = sentences
        for k in range(self.bertlayers):
            x = self.modelList[k](x, maskMatrix)
        return x

''' TransformerDecoder '''

# 下面两个代码块是针对decoder中的交叉注意力QKV写的
# 交叉注意力机制
class crossQKVAttention(nn.Module):
    def __init__(self, encoderOutdim, inputDim, crossQkvOutdim, dropOut):
        """
        :param encoderOutdim: encoder最终输出字特征维度
        :param inputDim: decoder中上一个自注意力层输出的query特征维度
        :param crossQkvOutdim: decoder cross attention QKV矩阵中特征的维度
        """
        super().__init__()
        self.linearQ = nn.Sequential(
            nn.Dropout(dropOut),
            nn.Linear(inputDim, crossQkvOutdim),
        )
        self.linearK = nn.Sequential(
            nn.Dropout(dropOut),
            nn.Linear(encoderOutdim, crossQkvOutdim)
        )
        self.linearV = nn.Sequential(
            nn.Dropout(dropOut),
            nn.Linear(encoderOutdim, crossQkvOutdim)
        )

    def forward(self, selfattOut, encoderOut):
        # Q的计算是selfattOut，其余是encoderOut
        Q = self.linearQ(selfattOut)
        K = self.linearK(encoderOut)
        V = self.linearV(encoderOut)
        dimk = torch.tensor(K.shape[2], device=selfattOut.device)  # 系数
        K = torch.transpose(K, 1, 2).contiguous()  # batchSize * qkvOutdim * length
        attentionScore = torch.matmul(Q, K) / torch.sqrt(dimk)  # batchSize * length * length
        attentionScore = nn.Softmax(dim=2)(attentionScore)
        headResult = torch.matmul(attentionScore, V)
        return headResult

# 交叉注意力模块
class crossQKVBlock(nn.Module):
    def __init__(self, crossAttHeads, encoderOutdim, selfattOutdim, crossQkvOutdim, blockOutdim, dropOut):
        """
        :param crossAttHeads:  交叉注意力头数
        :param encoderOutdim:  encoder最终输出的字特征维度
        :param selfattOutdim:  自注意力最后输出的qury特征维度
        :param crossQkvOutdim: 交叉注意力QKV矩阵特征维度
        :param blockOutdim:    交叉注意力最终输出特征维度（其实是与input保持不变的）
        """
        super(crossQKVBlock, self).__init__()
        self.attentionHeads = crossAttHeads
        self.modelList = nn.ModuleList([])
        for k in range(self.attentionHeads):
            self.modelList.append(crossQKVAttention(encoderOutdim, selfattOutdim, crossQkvOutdim, dropOut))
        self.lastLinear = nn.Sequential(
            nn.Dropout(dropOut),
            nn.Linear(crossAttHeads * crossQkvOutdim, blockOutdim)
        )

    def forward(self, selfattOut, encoderOut):
        headResultList = []
        for k in range(self.attentionHeads):
            headResultList.append(self.modelList[k](selfattOut, encoderOut))
        result = torch.concat(headResultList, dim=2)
        y = self.lastLinear(result)
        return y

# decoderPart的基本组成单元，decoderBlock
class DecoderBlock(nn.Module):
    def __init__(self, inputDim, qkvOutdim, heads, encoderOutdim, crossAttHeads, crossQkvOutdim, lastLinearMiddle,
                 dropOut):
        """
        :param inputDim: 无论是selfAttention的输入还是crossAttention的 Q矩阵输入，都是inputdim维度
        :param qkvOutdim: 指selfAttention的QKV矩阵输出特征维度
        :param heads:     指selfAttention头数
        :param encoderOutdim:  BertPart的输出维度，和BertPart的输入维度是一致的
        :param crossAttHeads:   crossAttention的头数
        :param crossQkvOutdim:  crossQKV的输出维度
        :param lastLinearMiddle: 最后线性层的中间层维度
        """
        super(DecoderBlock, self).__init__()
        self.QKVBlock = QKVBlock(inputDim, qkvOutdim, heads, inputDim,
                                 dropOut)  # decoder用残差连接结构，所以QKV的blockOutdim 和 inputDim相同
        self.layerNorm1 = nn.LayerNorm(inputDim)
        self.crossQKVBlock = crossQKVBlock(crossAttHeads, encoderOutdim, inputDim, crossQkvOutdim,
                                           inputDim, dropOut)  # decoder用残差连接结构，所以QKV的blockOutdim 和 inputDim相同
        self.layerNorm2 = nn.LayerNorm(inputDim)
        self.lastLinear = nn.Sequential(
            nn.Dropout(dropOut),
            nn.Linear(inputDim, lastLinearMiddle),
            nn.ReLU(),
            nn.Dropout(dropOut),
            nn.Linear(lastLinearMiddle, inputDim),
        )
        self.layerNorm3 = nn.LayerNorm(inputDim)

    def forward(self, x, encoderOut, maskMatrix=None):
        # x bachtSize * numberofQuery(number of set elements) * querydim
        # 1)首先selfAttention+residual Connection + NormLayer
        selfAttentionOut = self.QKVBlock(x, maskMatrix)
        y = self.layerNorm1(selfAttentionOut + x)
        # 2）然后crossAttention + residual Connection + NormLayer
        crossAttentionOut = self.crossQKVBlock(y, encoderOut)
        y = self.layerNorm2(crossAttentionOut + y)
        # 3) 最后是线性层
        y = self.layerNorm3(self.lastLinear(y) + y)
        return y

# decoderPart不负责tripleQuery的获取
class DecoderPart(nn.Module):
    def __init__(self, decoderLayers, inputDim, qkvOutdim, heads, encoderOutdim, crossAttHeads, crossQkvOutdim,
                 lastLinearMiddle, dropOut):
        """
        :param decoderLayers:  decoderBlock的层数
        :param inputDim:       query向量的初始输入维度
        :param qkvOutdim:      decoder中self attention 的QKV输出维度
        :param heads:          decoder中self attention 的头数
        :param encoderOutdim:  encoder的输出特征维度
        :param crossAttHeads:  decoder中cross attention 的头数
        :param crossQkvOutdim: decoder中cross attention 的QKV输出维度
        :param lastLinearMiddle: decoderBlock中最后一个线性层的中间层输出维度
        """
        super(DecoderPart, self).__init__()
        self.decoderLayers = decoderLayers
        self.modelList = nn.ModuleList([])
        for k in range(self.decoderLayers):
            self.modelList.append(DecoderBlock(inputDim, qkvOutdim, heads, encoderOutdim, crossAttHeads, crossQkvOutdim,
                                               lastLinearMiddle, dropOut))

    def forward(self, tripleQuery, encoderOut, maskMatrix=None):
        # encoderOut: batchSize * length * dim
        # tripleQuery :batchSize * m * dim
        tempQuery = tripleQuery
        for model in self.modelList:
            tempQuery = model(tempQuery, encoderOut, maskMatrix)
        return tempQuery

# ''' 初代剩余寿命预测模型，带偶然误差和认知误差 '''
# class TransformerMCWithNoise(nn.Module):
#     def __init__(self, encoderParamDict, rawDim, embDrop, reDrop):
#         super().__init__()
#         # 2）定义encoder
#         self.encoder = EncoderPart(
#             bertLayers=encoderParamDict['bertLayers'],
#             inputDim=encoderParamDict['bertInputDim'],
#             qkvOutdim=encoderParamDict['bertQkvOutdim'],
#             heads=encoderParamDict['bertQkvHeads'],
#             lastLinearMiddle=encoderParamDict['bertLastLinearMiddle'],
#             dropRate=encoderParamDict['dropRate']
#         )
#         self.embedding = nn.Sequential(
#             nn.Dropout(embDrop),
#             nn.Linear(rawDim, encoderParamDict['bertInputDim']),
#         )
#         self.mR = nn.Sequential(
#             nn.Dropout(reDrop),
#             nn.Linear(encoderParamDict['bertInputDim'], 1)
#         )
#         self.stdR = nn.Sequential(
#             nn.Dropout(reDrop),
#             nn.Linear(encoderParamDict['bertInputDim'], 1)
#         )
#
#     def forward(self, x):
#         x = x.squeeze(1)
#         embedding = self.embedding(x)
#         length = embedding.shape[1]
#         dim = embedding.shape[2]
#         encoderPositionEmbedding = self.getPositionEmbedding(length, dim).to(device=x.device)
#
#         # 1)获取Encoder的输出,encoder的输出获取一次就可以了，避免重复计算
#         encoderInput = embedding + encoderPositionEmbedding
#         encoderOut = self.encoder(encoderInput)
#         encoderOut = torch.mean(encoderOut, dim=1)
#         rul = self.mR(encoderOut)
#         std = self.stdR(encoderOut)
#         std = torch.log(1 + torch.exp(std))
#         return rul, std
#
#     def getPositionEmbedding(self, seqLen, dimension):
#         """
#         获取位置嵌入
#         :param seq_len: 序列长度
#         :param d_model: 嵌入维度
#         :return: 位置嵌入矩阵
#         """
#         # 创建一个位置矩阵，形状为 (seq_len, d_model)
#         position = torch.arange(seqLen, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
#         div_term = torch.exp(torch.arange(0, dimension, 2).float() * -(math.log(10000.0) / dimension))  # (d_model/2,)
#
#         # 计算位置嵌入
#         pos_embedding = torch.zeros(seqLen, dimension)
#         pos_embedding[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
#         pos_embedding[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
#
#         return pos_embedding


class ImprovedTransformer_2(nn.Module):
    def __init__(self, encoderParamDict, decoderParamDict, predictLength, rawDim, embDrop, reDrop, predictDim):
        super(ImprovedTransformer_2,self).__init__()
        # 2）定义encoder
        self.encoder = EncoderPart(
            bertLayers=encoderParamDict['bertLayers'],
            inputDim=encoderParamDict['bertInputDim'],
            qkvOutdim=encoderParamDict['bertQkvOutdim'],
            heads=encoderParamDict['bertQkvHeads'],
            lastLinearMiddle=encoderParamDict['bertLastLinearMiddle'],
            dropRate=encoderParamDict['dropRate']
        )
        self.embedding = nn.Sequential(
            nn.Dropout(embDrop),
            nn.Linear(rawDim, encoderParamDict['bertInputDim']),
        )
        self.mR = nn.Sequential(
            nn.Dropout(reDrop),
            nn.Linear(encoderParamDict['bertInputDim'], predictDim)
        )

        self.classifier = nn.Linear(encoderParamDict['bertInputDim'], 4)

        self.decoder = DecoderPart(
            decoderLayers=decoderParamDict['decoderLayers'],
            inputDim=decoderParamDict['decoderInputDim'],
            qkvOutdim=decoderParamDict['decoderQkvOutdim'],
            heads=decoderParamDict['decoderQkvHeads'],
            encoderOutdim=encoderParamDict['bertInputDim'],
            crossAttHeads=decoderParamDict['decoderCrossAttHeads'],
            crossQkvOutdim=decoderParamDict['decoderCrossAttOutdim'],
            lastLinearMiddle=decoderParamDict['decoderLastLinearMiddle'],
            dropOut=decoderParamDict['dropRate']
        )
        self.learnableQuery = nn.Parameter(torch.randn(predictLength, encoderParamDict['bertInputDim']))


    def forward(self, x):
        batchSize = x.shape[0]
        x = x.squeeze(1)
        embedding = self.embedding(x)
        length = embedding.shape[1]
        dim = embedding.shape[2]
        encoderPositionEmbedding = self.getPositionEmbedding(length, dim).to(device=x.device)

        # 1)获取Encoder的输出,encoder的输出获取一次就可以了，避免重复计算
        encoderInput = embedding + encoderPositionEmbedding
        encoderOut = self.encoder(encoderInput)

        # encoderOutForClassification = torch.mean(encoderOut, dim=1)


        # 2)获取decoder的
        length = self.learnableQuery.shape[0]
        dim = self.learnableQuery.shape[1]
        query = self.learnableQuery.unsqueeze(0).repeat(batchSize, 1, 1)
        decoderPositionEmbedding = self.getPositionEmbedding(length, dim).to(device=x.device)
        query = query + decoderPositionEmbedding
        decoderOut = self.decoder(query, encoderOut)
        rul = self.mR(decoderOut)
        encoderOutForClassification = torch.mean(decoderOut, dim=1)
        clss = self.classifier(encoderOutForClassification)
        return rul, clss

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

    def get_model_name(self):
        return 'iTransformer_3'