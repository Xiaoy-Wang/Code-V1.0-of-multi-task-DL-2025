import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from trainTest.early_stopping import EarlyStopping
from trainTest.get_lr_scheduler import LrScheduler
from trainTest.early_stopping import EarlyStopping

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
        :param inputDim:       qury向量的初始输入维度
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


''' 初代剩余寿命预测模型，带偶然误差和认知误差 '''


class TransformerMCWithNoise(nn.Module):
    def __init__(self, encoderParamDict, rawDim, embDrop, reDrop):
        super().__init__()
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
            nn.Linear(encoderParamDict['bertInputDim'], 1)
        )
        self.stdR = nn.Sequential(
            nn.Dropout(reDrop),
            nn.Linear(encoderParamDict['bertInputDim'], 1)
        )

    def forward(self, x):
        x = x.squeeze(1)
        embedding = self.embedding(x)
        length = embedding.shape[1]
        dim = embedding.shape[2]
        encoderPositionEmbedding = self.getPositionEmbedding(length, dim).to(device=x.device)

        # 1)获取Encoder的输出,encoder的输出获取一次就可以了，避免重复计算
        encoderInput = embedding + encoderPositionEmbedding
        encoderOut = self.encoder(encoderInput)
        encoderOut = torch.mean(encoderOut, dim=1)
        rul = self.mR(encoderOut)
        std = self.stdR(encoderOut)
        std = torch.log(1 + torch.exp(std))
        return rul, std

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


class TransformerMCWithNoiseForIndex(nn.Module):
    def __init__(self, encoderParamDict, decoderParamDict, predictLength, rawDim, embDrop, reDrop, predictDim):
        super().__init__()
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

        encoderOutForClassification = torch.mean(encoderOut, dim=1)

        # 2)获取decoder的
        length = self.learnableQuery.shape[0]
        dim = self.learnableQuery.shape[1]
        query = self.learnableQuery.unsqueeze(0).repeat(batchSize, 1, 1)
        decoderPositionEmbedding = self.getPositionEmbedding(length, dim).to(device=x.device)
        query = query + decoderPositionEmbedding
        decoderOut = self.decoder(query, encoderOut)

        rul = self.mR(decoderOut)

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
        return 'Transformer'

# %%
Epoch =100
from trainTest.optimizers.get_optimizer import Optimizer
def trainModel(path,current_exp_time,model, subject,  trainLoader, testLoader,validLoader,callbacks):
    early_stopping, scheduler, history_save_pic_name = None, None, None
    model_name = model.get_model_name()
    model_save_name =  f"./Results/{model_name}/{subject}/model_{current_exp_time}.pt"
    optimizer = Optimizer(model, optimizer_type=callbacks['optimizer'], lr=callbacks['initial_lr']).get_optimizer()
    if callbacks['lr_scheduler']['scheduler_type'] == 'None':
        print('不使用学习率调度器：')
    else:
        print('使用学习率调度器：', callbacks['lr_scheduler']['scheduler_type'])
        scheduler = LrScheduler(optimizer, callbacks['lr_scheduler']['scheduler_type'],
                                callbacks['lr_scheduler']['params'], callbacks['epoch']).get_scheduler()
    if callbacks['early_stopping']['use_es']:
        print('使用早停：')
        early_stopping = EarlyStopping(patience=callbacks['early_stopping']['params']['patience'],
                                       verbose=callbacks['early_stopping']['params']['verbose'],
                                       delta=callbacks['early_stopping']['params']['delta'],
                                       path=model_save_name)

    model.cuda()
    model.double()

    # 初始化训练和测试指标列表
    trainRmseLossEpoch = []
    trainCrossLossEpoch = []
    trainAccuracyEpoch = []
    trainR2ScoreEpoch = []

    validRmseLossEpoch = []
    validCrossLossEpoch = []
    validAccuracyEpoch = []
    validR2ScoreEpoch = []

    testRmseLossEpoch = []
    testCrossLossEpoch = []
    testAccuracyEpoch = []
    testR2ScoreEpoch = []
    testF1ScoreEpoch = []
    testRecallEpoch = []
    testRMSEEpoch = []

    # 模型训练

    for e in range(Epoch):
        model.train()
        tempRmseLoss = 0
        tempCrossLoss = 0
        tempAccuracy = 0
        tempR2Score = 0

        for data, label1, label2 in trainLoader:
            data = data.to(dtype=torch.double, device='cuda')
            label1 = label1.to(dtype=torch.double, device='cuda')
            label2 = label2.to(device='cuda').view(-1)

            # 前向传播
            preResult, cResult = model(data)

            # 计算损失
            loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
            loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失

            # 总损失
            loss = loss1 + loss2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计各项指标
            tempRmseLoss += loss1.detach().to(device='cpu').item()
            tempCrossLoss += loss2.detach().to(device='cpu').item()

            # 相位预测准确度
            maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
            label2_cpu = label2.to(device='cpu').view(-1, 1)
            tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]

            # 角度预测R²得分
            label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
            preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
            tempR2Score += r2_score(label1_flat, preResult_flat)

        # 计算每个epoch的平均值
        tempRmseLoss /= len(trainLoader)
        tempCrossLoss /= len(trainLoader)
        tempAccuracy /= len(trainLoader)
        tempR2Score /= len(trainLoader)

        # 存储每个epoch的结果
        trainRmseLossEpoch.append(tempRmseLoss)
        trainCrossLossEpoch.append(tempCrossLoss)
        trainAccuracyEpoch.append(tempAccuracy)
        trainR2ScoreEpoch.append(tempR2Score)

            # 进入验证阶段
        model.eval()
        with torch.no_grad():
            # 用于计算损失和指标的临时变量
            tempRmseLoss = 0
            tempCrossLoss = 0
            tempAccuracy = 0
            tempR2Score = 0
            for data, label1, label2 in validLoader:
                data = data.to(dtype=torch.double, device='cuda')
                label1 = label1.to(dtype=torch.double, device='cuda')
                label2 = label2.to(device='cuda').view(-1)

                # 前向传播
                preResult, cResult = model(data)

                # 计算损失
                loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
                loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失

                # 累计各项指标
                tempRmseLoss += loss1.detach().to(device='cpu').item()
                tempCrossLoss += loss2.detach().to(device='cpu').item()

                # 相位预测准确度
                maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
                label2_cpu = label2.to(device='cpu').view(-1, 1)
                tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]

                # 角度预测R²得分
                label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
                preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
                tempR2Score += r2_score(label1_flat, preResult_flat)

            # 计算每个epoch的平均值
            tempRmseLoss /= len(validLoader)
            tempCrossLoss /= len(validLoader)
            tempAccuracy /= len(validLoader)
            tempR2Score /= len(validLoader)

        # 存储每个epoch的结果
        validRmseLossEpoch.append(tempRmseLoss)
        validCrossLossEpoch.append(tempCrossLoss)
        validAccuracyEpoch.append(tempAccuracy)
        validR2ScoreEpoch.append(tempR2Score)
        if callbacks['lr_scheduler']['scheduler_type'] in ['StepLR', 'MultiStepLR', 'AutoWarmupLR',
                                                           'GradualWarmupLR',
                                                           'ExponentialLR']:
            scheduler.step()
        elif callbacks['lr_scheduler']['scheduler_type'] == 'ReduceLROnPlateau':
            # scheduler.step(acc_valid_epoch / 100)
            scheduler.step(tempAccuracy+tempR2Score, model)
        else:
            pass
        if callbacks['early_stopping']['use_es']:
            # 判断早停
            early_stopping(tempRmseLoss+tempCrossLoss, model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping，保存模型：")
                break
            # 7. 如果不使用早停中的模型保存，则在模型训练结束后保存模型
            # 如果文件夹路径不存在，就创建它
    save_dir = os.path.dirname(model_save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"目录 {save_dir} 已创建。")
    if not callbacks['early_stopping']['use_es']:
        print('保存最后一次训练后的模型：')
        torch.save(model, model_save_name)
    model = torch.load(model_save_name)
    ##测试集
    model.eval()
    with torch.no_grad():
        tempRmseLoss = 0
        tempCrossLoss = 0
        tempAccuracy = 0
        tempR2Score = 0
        tempF1Score = 0
        tempRecall = 0
        tempRMSE = 0

        for data, label1, label2 in testLoader:
            data = data.to(dtype=torch.double, device='cuda')
            label1 = label1.to(dtype=torch.double, device='cuda')
            label2 = label2.to(device='cuda').view(-1)

            # 前向传播
            preResult, cResult = model(data)

            # 计算损失
            loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
            loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失

            # 累计各项指标
            tempRmseLoss += loss1.detach().to(device='cpu').item()
            tempCrossLoss += loss2.detach().to(device='cpu').item()

            # 相位预测准确度
            maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
            label2_cpu = label2.to(device='cpu').view(-1, 1)
            tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]
            tempF1Score += f1_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro')
            tempRecall += recall_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro', zero_division=0)
            # 角度预测R²得分
            label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
            preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
            tempR2Score += r2_score(label1_flat, preResult_flat)
            aa = mean_squared_error(label1_flat, preResult_flat)
            tempRMSE += np.sqrt(aa)


            # 计算每个epoch的平均值
        tempRmseLoss /= len(testLoader)
        tempCrossLoss /= len(testLoader)
        tempAccuracy /=  len(testLoader)
        tempR2Score /= len(testLoader)
        tempF1Score  /= len(testLoader)
        tempRecall /= len(testLoader)
        tempRMSE /= len(testLoader)



        # 存储每个epoch的结果
        testRmseLossEpoch.append(tempRmseLoss)
        testCrossLossEpoch.append(tempCrossLoss)
        testAccuracyEpoch.append(tempAccuracy)
        testR2ScoreEpoch.append(tempR2Score)
        testF1ScoreEpoch.append(tempF1Score)
        testRecallEpoch.append(tempRecall)
        testRMSEEpoch.append(tempRMSE)
        # 打印当前epoch的测试结果
        print(f'Epoch: {e + 1}/{Epoch} | '
              f'Test Loss: {tempRmseLoss:.4f} | '
              f'Test F1 score: {tempF1Score:.4f} | '
              f'Test Recall: {tempRecall:.4f} | '
              f'Test Accuracy: {tempAccuracy:.4f} | '
              f'Test R² Score: {tempR2Score:.4f}')

    # 转换为numpy数组
    trainRmseLossEpoch = np.array(trainRmseLossEpoch)
    trainCrossLossEpoch = np.array(trainCrossLossEpoch)
    trainAccuracyEpoch = np.array(trainAccuracyEpoch)
    trainR2ScoreEpoch = np.array(trainR2ScoreEpoch)

    validRmseLossEpoch = np.array(validRmseLossEpoch)
    validCrossLossEpoch = np.array(validCrossLossEpoch)
    validAccuracyEpoch = np.array(validAccuracyEpoch)
    validR2ScoreEpoch = np.array(validR2ScoreEpoch)

    testRmseLossEpoch = np.array(testRmseLossEpoch)
    testCrossLossEpoch = np.array(testCrossLossEpoch)
    testAccuracyEpoch = np.array(testAccuracyEpoch)
    testR2ScoreEpoch = np.array(testR2ScoreEpoch)
    testF1ScoreEpoch = np.array(testF1ScoreEpoch)
    testRecallEpoch = np.array(testRecallEpoch)
    testRMSEEpoch = np.array(testRMSEEpoch)
    # 构建 test_metrics DataFrame
    test_metrics = pd.DataFrame({
        'Test Accuracy': [round(testAccuracyEpoch[-1],5)],
        'Test F1 Score': [round(testF1ScoreEpoch[-1],5)],
        'Test Recall': [round(testRecallEpoch[-1], 5)],
        'Test MSE': [round(testRmseLossEpoch[-1],4)],
        'Test R² Score': [round(testR2ScoreEpoch[-1], 5)],
        'Test RMSE': [round(testRMSEEpoch[-1], 5)]
    })
    # 指定保存路径
    test_metrics_save_path = os.path.join(path, f'test_metrics_{current_exp_time}.csv')
    # test_metrics_save_path = f"./Results/{model_name}/{subject}/test_metrics_{current_exp_time}.csv"
    test_metrics.to_csv(test_metrics_save_path, index=False)

    #
    tain_results = pd.DataFrame({
        'Epoch': np.arange(1, len(trainRmseLossEpoch) + 1),
        'train MSE Loss': trainRmseLossEpoch,
        'train Cross Entropy Loss': trainCrossLossEpoch,
        'train Accuracy': trainAccuracyEpoch,
        'train R² Score': trainR2ScoreEpoch
    })
    valid_results = pd.DataFrame({
        'Epoch': np.arange(1, len(validRmseLossEpoch) + 1),
        'valid MSE Loss': validRmseLossEpoch,
        'valid Cross Entropy Loss': validCrossLossEpoch,
        'valid Accuracy': validAccuracyEpoch,
        'valid R² Score': validR2ScoreEpoch
    })

    test_results = pd.DataFrame({
        'Epoch': np.arange(1, len(testRmseLossEpoch) + 1),
        'test MSE Loss': testRmseLossEpoch,
        'test Cross Entropy Loss': testCrossLossEpoch,
        'test Accuracy': testAccuracyEpoch,
        'test R² Score': testR2ScoreEpoch,
        'Test RMSE': testRMSEEpoch
    })

    train_results_save_path = os.path.join(path, f"train_process_{current_exp_time}.csv")
    print(train_results_save_path)
    valid_results_save_path = os.path.join(path, f"valid_process_{current_exp_time}.csv")
    test_results_save_path = os.path.join(path, f"test_process_{current_exp_time}.csv")

    tain_results.to_csv(train_results_save_path, index=False)
    print('111')
    valid_results.to_csv(valid_results_save_path, index=False)
    test_results.to_csv(test_results_save_path, index=False)

    return (trainRmseLossEpoch, trainCrossLossEpoch, trainAccuracyEpoch, trainR2ScoreEpoch,
            testRmseLossEpoch, testCrossLossEpoch, testAccuracyEpoch, testR2ScoreEpoch)





def trainModel_CNN(path,current_exp_time,model, Epoch, optimizer, trainLoader, testLoader,validLoader,EarlyStop=False):
    model_name = model.get_model_name()
    model_save_name = os.path.join(path, f'model_{current_exp_time}.pt')
    # model_save_name =  f"./Results/{model_name}/{subject}/model_{current_exp_time}.pt"
    model.cuda()
    model.double()

    # 初始化训练和测试指标列表
    trainRmseLossEpoch = []
    trainCrossLossEpoch = []
    trainAccuracyEpoch = []
    trainR2ScoreEpoch = []

    validRmseLossEpoch = []
    validCrossLossEpoch = []
    validAccuracyEpoch = []
    validR2ScoreEpoch = []

    testRmseLossEpoch = []
    testCrossLossEpoch = []
    testAccuracyEpoch = []
    testR2ScoreEpoch = []
    testF1ScoreEpoch = []
    testRecallEpoch = []

    # 模型训练
    for e in range(Epoch):
        model.train()
        tempRmseLoss = 0
        tempCrossLoss = 0
        tempAccuracy = 0
        tempR2Score = 0

        for data, label1, label2 in trainLoader:
            data = data.unsqueeze(1)
            data = data.to(dtype=torch.double, device='cuda')
            label1 = label1.to(dtype=torch.double, device='cuda')
            label2 = label2.to(device='cuda').view(-1)

            # 前向传播
            preResult, cResult = model(data)

            # 计算损失
            loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
            loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失

            # 总损失
            loss = loss1 + loss2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计各项指标
            tempRmseLoss += loss1.detach().to(device='cpu').item()
            tempCrossLoss += loss2.detach().to(device='cpu').item()

            # 相位预测准确度
            maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
            label2_cpu = label2.to(device='cpu').view(-1, 1)
            tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]

            # 角度预测R²得分
            label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
            preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
            tempR2Score += r2_score(label1_flat, preResult_flat)

        # 计算每个epoch的平均值
        tempRmseLoss /= len(trainLoader)
        tempCrossLoss /= len(trainLoader)
        tempAccuracy /= len(trainLoader)
        tempR2Score /= len(trainLoader)

        # 存储每个epoch的结果
        trainRmseLossEpoch.append(tempRmseLoss)
        trainCrossLossEpoch.append(tempCrossLoss)
        trainAccuracyEpoch.append(tempAccuracy)
        trainR2ScoreEpoch.append(tempR2Score)
        # 验证阶段
        if EarlyStop:
            # print('使用早停：')
            earlystop = EarlyStopping(path=model_save_name,patience=10, verbose=1)
            # 进入验证阶段
            model.eval()
            with torch.no_grad():
                # 用于计算损失和指标的临时变量
                tempRmseLoss = 0
                tempCrossLoss = 0
                tempAccuracy = 0
                tempR2Score = 0
                for data, label1, label2 in validLoader:
                    data = data.unsqueeze(1)
                    data = data.to(dtype=torch.double, device='cuda')
                    label1 = label1.to(dtype=torch.double, device='cuda')
                    label2 = label2.to(device='cuda').view(-1)

                    # 前向传播
                    preResult, cResult = model(data)

                    # 计算损失
                    loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
                    loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失

                    # 累计各项指标
                    tempRmseLoss += loss1.detach().to(device='cpu').item()
                    tempCrossLoss += loss2.detach().to(device='cpu').item()

                    # 相位预测准确度
                    maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
                    label2_cpu = label2.to(device='cpu').view(-1, 1)
                    tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]

                    # 角度预测R²得分
                    label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
                    preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
                    tempR2Score += r2_score(label1_flat, preResult_flat)

                # 计算每个epoch的平均值
                tempRmseLoss /= len(validLoader)
                tempCrossLoss /= len(validLoader)
                tempAccuracy /= len(validLoader)
                tempR2Score /= len(validLoader)

                # 存储每个epoch的结果
                validRmseLossEpoch.append(tempRmseLoss)
                validCrossLossEpoch.append(tempCrossLoss)
                validAccuracyEpoch.append(tempAccuracy)
                validR2ScoreEpoch.append(tempR2Score)

            # 通过earlystop判断是否需要早停
            earlystop(tempRmseLoss+tempCrossLoss,model)  # 传入当前epoch的验证R²得分

            # 如果需要早停，则退出训练
            if earlystop.early_stop:
                print("Early stopping triggered.")
                break
            # 7. 如果不使用早停中的模型保存，则在模型训练结束后保存模型
            # 如果文件夹路径不存在，就创建它
    save_dir = os.path.dirname(model_save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"目录 {save_dir} 已创建。")
    if not EarlyStop:
        print('保存最后一次训练后的模型：')
        torch.save(model, model_save_name)
    model = torch.load(model_save_name)
    ##测试集
    model.eval()
    with torch.no_grad():
        tempRmseLoss = 0
        tempCrossLoss = 0
        tempAccuracy = 0
        tempR2Score = 0
        tempF1Score = 0
        tempRecall = 0


        for data, label1, label2 in testLoader:
            data = data.unsqueeze(1)
            data = data.to(dtype=torch.double, device='cuda')
            label1 = label1.to(dtype=torch.double, device='cuda')
            label2 = label2.to(device='cuda').view(-1)

            # 前向传播
            preResult, cResult = model(data)

            # 计算损失
            loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
            loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失

            # 累计各项指标
            tempRmseLoss += loss1.detach().to(device='cpu').item()
            tempCrossLoss += loss2.detach().to(device='cpu').item()

            # 相位预测准确度
            maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
            label2_cpu = label2.to(device='cpu').view(-1, 1)
            tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]
            tempF1Score += f1_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro')
            tempRecall += recall_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro', zero_division=0)
            # 角度预测R²得分
            label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
            preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
            tempR2Score += r2_score(label1_flat, preResult_flat)


        # 计算每个epoch的平均值
        tempRmseLoss = tempRmseLoss / len(testLoader)
        tempCrossLoss = tempCrossLoss / len(testLoader)
        tempAccuracy = tempAccuracy / len(testLoader)
        tempR2Score /= len(testLoader)
        tempF1Score  /= len(testLoader)
        tempRecall /= len(testLoader)


        # 存储每个epoch的结果
        testRmseLossEpoch.append(tempRmseLoss)
        testCrossLossEpoch.append(tempCrossLoss)
        testAccuracyEpoch.append(tempAccuracy)
        testR2ScoreEpoch.append(tempR2Score)
        testF1ScoreEpoch.append(tempF1Score)
        testRecallEpoch.append(tempRecall)
        # 打印当前epoch的测试结果
        print(f'Epoch: {e + 1}/{Epoch} | '
              f'Test Loss: {tempRmseLoss:.4f} | '
              f'Test F1 score: {tempF1Score:.4f} | '
              f'Test Recall: {tempRecall:.4f} | '
              f'Test Accuracy: {tempAccuracy:.4f} | '
              f'Test R² Score: {tempR2Score:.4f}')

    # 转换为numpy数组
    trainRmseLossEpoch = np.array(trainRmseLossEpoch)
    trainCrossLossEpoch = np.array(trainCrossLossEpoch)
    trainAccuracyEpoch = np.array(trainAccuracyEpoch)
    trainR2ScoreEpoch = np.array(trainR2ScoreEpoch)

    validRmseLossEpoch = np.array(validRmseLossEpoch)
    validCrossLossEpoch = np.array(validCrossLossEpoch)
    validAccuracyEpoch = np.array(validAccuracyEpoch)
    validR2ScoreEpoch = np.array(validR2ScoreEpoch)

    testRmseLossEpoch = np.array(testRmseLossEpoch)
    testCrossLossEpoch = np.array(testCrossLossEpoch)
    testAccuracyEpoch = np.array(testAccuracyEpoch)
    testR2ScoreEpoch = np.array(testR2ScoreEpoch)
    testF1ScoreEpoch = np.array(testF1ScoreEpoch)
    testRecallEpoch = np.array(testRecallEpoch)

    # 构建 test_metrics DataFrame
    test_metrics = pd.DataFrame({
        'Test Accuracy': [round(testAccuracyEpoch[-1],5)],
        'Test F1 Score': [round(testF1ScoreEpoch[-1],5)],
        'Test Recall': [round(testRecallEpoch[-1], 5)],
        'Test MSE': [round(testRmseLossEpoch[-1],4)],
        'Test R² Score': [round(testR2ScoreEpoch[-1], 5)],
    })
    # 指定保存路径
    test_metrics_save_path = os.path.join(path, f'test_metrics_{current_exp_time}.csv')
    # test_metrics_save_path = f"./Results/{model_name}/{subject}/test_metrics_{current_exp_time}.csv"
    test_metrics.to_csv(test_metrics_save_path, index=False)

    #
    tain_results = pd.DataFrame({
        'Epoch': np.arange(1, len(trainRmseLossEpoch) + 1),
        'train MSE Loss': trainRmseLossEpoch,
        'train Cross Entropy Loss': trainCrossLossEpoch,
        'train Accuracy': trainAccuracyEpoch,
        'train R² Score': trainR2ScoreEpoch
    })
    valid_results = pd.DataFrame({
        'Epoch': np.arange(1, len(trainRmseLossEpoch) + 1),
        'valid MSE Loss': validRmseLossEpoch,
        'valid Cross Entropy Loss': validCrossLossEpoch,
        'valid Accuracy': validAccuracyEpoch,
        'valid R² Score': validR2ScoreEpoch
    })

    test_results = pd.DataFrame({
        'Epoch': np.arange(1, len(trainRmseLossEpoch) + 1),
        'test MSE Loss': testRmseLossEpoch,
        'test Cross Entropy Loss': testCrossLossEpoch,
        'test Accuracy': testAccuracyEpoch,
        'test R² Score': testR2ScoreEpoch
    })

    train_results_save_path =os.path.join(path, f"train_process_{current_exp_time}.csv")
    valid_results_save_path = os.path.join(path, f"valid_process_{current_exp_time}.csv")
    test_results_save_path = os.path.join(path, f"test_process_{current_exp_time}.csv")

    tain_results.to_csv(train_results_save_path, index=False)
    valid_results.to_csv(valid_results_save_path, index=False)
    test_results.to_csv(test_results_save_path, index=False)
    return (trainRmseLossEpoch, trainCrossLossEpoch, trainAccuracyEpoch, trainR2ScoreEpoch,
            testRmseLossEpoch, testCrossLossEpoch, testAccuracyEpoch, testR2ScoreEpoch)
# %
# import time
# def trainModel(subject,current_exp_time,model, Epoch, optimizer, trainLoader, testLoader,validLoader,EarlyStop=False):
#
#     model_save_name =  f"./Results/{subject}/model_{current_exp_time}.pt"
#     model.cuda()
#     model.double()
#
#     # 初始化训练和测试指标列表
#     trainRmseLossEpoch = []
#     trainCrossLossEpoch = []
#     trainAccuracyEpoch = []
#     trainR2ScoreEpoch = []
#     trainF1ScoreEpoch = []
#     trainRMSEEpoch = []
#     trainRecallEpoch = []
#
#     testRmseLossEpoch = []
#     testCrossLossEpoch = []
#     testAccuracyEpoch = []
#     testR2ScoreEpoch = []
#     testF1ScoreEpoch = []
#     testRMSEEpoch = []
#     testRecallEpoch = []
#
#     validRmseLossEpoch = []
#     validCrossLossEpoch = []
#     validAccuracyEpoch = []
#     validR2ScoreEpoch = []
#     validF1ScoreEpoch = []
#     validRMSEEpoch = []
#     validRecallEpoch = []
#
#     print('开始训练：')
#     start = time.time()
#     # 模型训练
#     for e in range(Epoch):
#         model.train()
#         tempRmseLoss = 0
#         tempCrossLoss = 0
#         tempAccuracy = 0
#         tempF1Score = 0
#         tempR2Score = 0
#         tempRMSE = 0
#         tempRecall = 0
#
#         for data, label1, label2 in trainLoader:
#             data = data.to(dtype=torch.double, device='cuda')
#             label1 = label1.to(dtype=torch.double, device='cuda')
#             label2 = label2.to(device='cuda').view(-1)
#
#             # 前向传播
#             preResult, cResult = model(data)
#
#             # 计算损失
#             loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
#             loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失
#
#             # 总损失
#             loss = loss1 + loss2
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             # 累计各项指标
#             tempRmseLoss += loss1.detach().to(device='cpu').item()
#             tempCrossLoss += loss2.detach().to(device='cpu').item()
#
#             # 相位预测准确度
#             maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
#             label2_cpu = label2.to(device='cpu').view(-1, 1)
#             tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]
#             tempF1Score += f1_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro')
#             tempRecall += recall_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro', zero_division=0)
#             # 角度预测R²得分
#             label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
#             preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
#             tempR2Score += r2_score(label1_flat, preResult_flat)
#             tempRMSE += root_mean_squared_error(label1_flat, preResult_flat)
#
#
#         # 计算每个epoch的平均值
#         tempRmseLoss /= len(trainLoader)
#         tempCrossLoss /= len(trainLoader)
#         tempAccuracy /= len(trainLoader)
#         tempF1Score /= len(trainLoader)
#         tempR2Score /= len(trainLoader)
#         tempRMSE /= len(trainLoader)
#         tempRecall /= len(trainLoader)
#
#         # 存储每个epoch的结果
#         trainRmseLossEpoch.append(tempRmseLoss)
#         trainCrossLossEpoch.append(tempCrossLoss)
#         trainAccuracyEpoch.append(tempAccuracy)
#         trainF1ScoreEpoch.append(tempF1Score)
#         trainR2ScoreEpoch.append(tempR2Score)
#         trainRMSEEpoch.append(tempRMSE)
#         trainRecallEpoch.append(tempRecall)
#         # 验证阶段
#         if EarlyStop:
#             # print('使用早停：')
#             earlystop = EarlyStopping(path=model_save_name,patience=10, verbose=1)
#             # 进入验证阶段
#             model.eval()
#             with torch.no_grad():
#                 # 用于计算损失和指标的临时变量
#                 tempRmseLoss = 0
#                 tempCrossLoss = 0
#                 tempAccuracy = 0
#                 tempF1Score = 0
#                 tempR2Score = 0
#                 tempRMSE = 0
#                 tempRecall = 0
#                 for data, label1, label2 in validLoader:
#                     data = data.to(dtype=torch.double, device='cuda')
#                     label1 = label1.to(dtype=torch.double, device='cuda')
#                     label2 = label2.to(device='cuda').view(-1)
#
#                     # 前向传播
#                     preResult, cResult = model(data)
#
#                     # 计算损失
#                     loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
#                     loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失
#
#                     # 累计各项指标
#                     tempRmseLoss += loss1.detach().to(device='cpu').item()
#                     tempCrossLoss += loss2.detach().to(device='cpu').item()
#
#                     # 相位预测准确度
#                     maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
#                     label2_cpu = label2.to(device='cpu').view(-1, 1)
#                     tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]
#                     tempF1Score += f1_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro')
#                     tempRecall += recall_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro', zero_division=0)
#                     # 角度预测R²得分
#                     label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
#                     preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
#                     tempR2Score += r2_score(label1_flat, preResult_flat)
#                     tempRMSE += root_mean_squared_error(label1_flat, preResult_flat)
#
#                     # 计算每个epoch的平均值
#                 tempRmseLoss /= len(trainLoader)
#                 tempCrossLoss /= len(trainLoader)
#                 tempAccuracy /= len(trainLoader)
#                 tempF1Score /= len(trainLoader)
#                 tempR2Score /= len(trainLoader)
#                 tempRMSE /= len(trainLoader)
#
#                 # 存储每个epoch的结果
#                 validRmseLossEpoch.append(tempRmseLoss)
#                 validCrossLossEpoch.append(tempCrossLoss)
#                 validAccuracyEpoch.append(tempAccuracy)
#                 validR2ScoreEpoch.append(tempR2Score)
#                 validF1ScoreEpoch.append(tempF1Score)
#                 validRMSEEpoch.append(tempRMSE)
#             # 通过earlystop判断是否需要早停
#             earlystop(tempAccuracy+tempR2Score,model)  # 传入当前epoch的验证R²得分
#
#             # 如果需要早停，则退出训练
#             if earlystop.early_stop:
#                 print("Early stopping triggered.")
#                 break
#             # 7. 如果不使用早停中的模型保存，则在模型训练结束后保存模型
#             # 如果文件夹路径不存在，就创建它
#     # end = time.time()
#     # train_time = end - start
#     # print('训练完成，耗时： %.2f 分钟' % (float(train_time) / 60.0))
#     # save_dir = os.path.dirname(model_save_name)
#     # if not os.path.exists(save_dir):
#     #     os.makedirs(save_dir)
#     #     print(f"目录 {save_dir} 已创建。")
#     # if not EarlyStop:
#     #     print('保存最后一次训练后的模型：')
#     #     torch.save(model, model_save_name)
#     # # model = torch.load(model_save_name)
#         ##测试集
#         model.eval()
#         with torch.no_grad():
#             tempRmseLoss = 0
#             tempCrossLoss = 0
#             tempAccuracy = 0
#             tempF1Score = 0
#             tempR2Score = 0
#             tempRMSE = 0
#             tempRecall = 0
#             for data, label1, label2 in testLoader:
#                 data = data.to(dtype=torch.double, device='cuda')
#                 label1 = label1.to(dtype=torch.double, device='cuda')
#                 label2 = label2.to(device='cuda').view(-1)
#
#                 # 前向传播
#                 preResult, cResult = model(data)
#
#                 # 计算损失
#                 loss1 = nn.MSELoss()(preResult, label1)  # 角度预测的MSE损失
#                 loss2 = nn.CrossEntropyLoss()(cResult, label2)  # 相位预测的交叉熵损失
#
#                 # 累计各项指标
#                 tempRmseLoss += loss1.detach().to(device='cpu').item()
#                 tempCrossLoss += loss2.detach().to(device='cpu').item()
#
#                 # 相位预测准确度
#                 maxIndex = torch.argmax(cResult.detach().to(device='cpu'), dim=1).view(-1, 1)
#                 label2_cpu = label2.to(device='cpu').view(-1, 1)
#                 tempAccuracy += torch.sum(label2_cpu == maxIndex) / label2_cpu.shape[0]
#                 tempF1Score += f1_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro')
#                 tempRecall += recall_score(label2_cpu.numpy(), maxIndex.numpy(), average='macro', zero_division=0)
#                 # 角度预测R²得分
#                 label1_flat = label1.detach().view(-1).to(device='cpu').numpy()
#                 preResult_flat = preResult.detach().view(-1).to(device='cpu').numpy()
#                 tempR2Score += r2_score(label1_flat, preResult_flat)
#                 tempRMSE += root_mean_squared_error(label1_flat, preResult_flat)
#
#                 # 计算每个epoch的平均值
#             tempRmseLoss /= len(trainLoader)
#             tempCrossLoss /= len(trainLoader)
#             tempAccuracy /= len(trainLoader)
#             tempF1Score /= len(trainLoader)
#             tempR2Score /= len(trainLoader)
#             tempRMSE /= len(trainLoader)
#             tempRecall /= len(trainLoader)
#
#             # 存储每个epoch的结果
#             testRmseLossEpoch.append(tempRmseLoss)
#             testCrossLossEpoch.append(tempCrossLoss)
#             testAccuracyEpoch.append(tempAccuracy)
#             testR2ScoreEpoch.append(tempR2Score)
#             testF1ScoreEpoch.append(tempF1Score)
#             testRMSEEpoch.append(tempRMSE)
#             testRecallEpoch.append(tempRecall)
#         # 打印当前epoch的测试结果
#         # 打印当前epoch的测试结果
#         print(f'Epoch: {e + 1}/{Epoch} | '
#               f'Test Loss: {tempRmseLoss:.4f} | '
#               f'Test F1 score: {tempF1Score:.4f} | '
#               f'Test Recall: {tempRecall:.4f} | '
#               f'Test Accuracy: {tempAccuracy:.4f} | '
#               f'Test RMSE: {tempRMSE:.4f} | '
#               f'Test R² Score: {tempR2Score:.4f}')
#
#     # 转换为numpy数组
#     trainRmseLossEpoch = np.array(trainRmseLossEpoch)
#     trainCrossLossEpoch = np.array(trainCrossLossEpoch)
#     trainAccuracyEpoch = np.array(trainAccuracyEpoch)
#     trainR2ScoreEpoch = np.array(trainR2ScoreEpoch)
#     trainF1ScoreEpoch = np.array(trainF1ScoreEpoch)
#     trainRMSEEpoch = np.array(trainRMSEEpoch)
#     trainRecallEpoch = np.array(trainRecallEpoch)
#
#     testRmseLossEpoch = np.array(testRmseLossEpoch)
#     testCrossLossEpoch = np.array(testCrossLossEpoch)
#     testAccuracyEpoch = np.array(testAccuracyEpoch)
#     testR2ScoreEpoch = np.array(testR2ScoreEpoch)
#     testF1ScoreEpoch = np.array(testF1ScoreEpoch)
#     testRMSEEpoch = np.array(testRMSEEpoch)
#     testRecallEpoch = np.array(testRecallEpoch)
#
#     # test_results = pd.DataFrame({
#     #     'Epoch': np.arange(1, len(testRmseLossEpoch) + 1),
#     #     'Test MSE Loss': testRmseLossEpoch,
#     #     'Test Cross Entropy Loss': testCrossLossEpoch,
#     #     'Test Accuracy': testAccuracyEpoch,
#     #     'Test R² Score': testR2ScoreEpoch
#     # })
#     # 构建 test_metrics DataFrame
#     test_metrics = pd.DataFrame({
#         'Test Accuracy': [round(testAccuracyEpoch[-1],5)],
#         'Test F1 Score': [round(testF1ScoreEpoch[-1],5)],
#         'Test Recall Score': [round(testRecallEpoch[-1], 5)],
#         'Test R² Score': [round(testR2ScoreEpoch[-1], 5)],
#         'Test RMSE ': [round(testRMSEEpoch[-1], 5)],
#     })
#     # 指定保存路径
#     test_metrics_save_path = f"./Results/{subject}/test_metrics_{current_exp_time}.csv"
#     # test_results_save_path = f"./Results/{subject}/test_results_{current_exp_time}.csv"
#     # 保存到 CSV 文件
#     # test_results.to_csv(test_results_save_path, index=False)
#     # 保存到 CSV 文件
#     test_metrics.to_csv(test_metrics_save_path, index=False)
#
#     return (trainRmseLossEpoch, trainCrossLossEpoch, trainAccuracyEpoch, trainR2ScoreEpoch,
#             testRmseLossEpoch, testCrossLossEpoch, testAccuracyEpoch, testR2ScoreEpoch)


# def trainModel(model, Epoch, optimizer, trainLoader, testLoader):
#     model.cuda()
#     model.double()
#     trainRmseLossEpoch = []
#     trainCrossLossEpoch = []
#     trainAccuracyEpoch = []
#     trainR2ScoreEpoch = []
#
#     testRmseLossEpoch = []
#     testCrossLossEpoch = []
#     testAccuracyEpoch = []
#     testR2ScoreEpoch = []
#     # 模型训练
#     for e in range(Epoch):
#         model.train()
#         tempRmseLoss = 0
#         tempCrossLoss = 0
#         tempAccuracy = 0
#         tempR2Score = 0
#
#         for data,label1,label2 in trainLoader:
#             data = data.to(dtype=torch.double, device='cuda')
#             label1 = label1.to(dtype=torch.double, device='cuda')
#             label2 = label2.to(device='cuda').view(-1)
#
#             preResult, cResult = model(data)
#
#             loss1 = nn.MSELoss()(preResult,label1)
#             loss2 = nn.CrossEntropyLoss()(cResult,label2)
#
#             loss = loss1 + loss2
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             tempRmseLoss += loss1.detach().to(device='cpu').item()
#             tempCrossLoss += loss2.detach().to(device='cpu').item()
#             maxIndex = torch.argmax(cResult.detach().to(device='cpu'),dim=1).view(-1,1)
#             label2 = label2.to(device='cpu').view(-1,1)
#             tempAccuracy += torch.sum(label2==maxIndex)/label2.shape[0]
#             tempR2Score = r2_score(preResult, label1)
#
#         tempRmseLoss = tempRmseLoss / len(trainLoader)
#         tempCrossLoss = tempCrossLoss / len(trainLoader)
#         tempAccuracy = tempAccuracy/len(trainLoader)
#
#         trainRmseLossEpoch.append(tempRmseLoss)
#         trainCrossLossEpoch.append(tempCrossLoss)
#         trainAccuracyEpoch.append(tempAccuracy.item())
#
#
#         model.eval()
#         tempRmseLoss = 0
#         tempCrossLoss = 0
#         tempAccuracy = 0
#         for data,label1,label2 in testLoader:
#             data = data.to(dtype=torch.double, device='cuda')
#             label1 = label1.to(dtype=torch.double, device='cuda')
#             label2 = label2.to(device='cuda').view(-1)
#
#             preResult, cResult = model(data)
#
#             loss1 = nn.MSELoss()(preResult,label1)
#             loss2 = nn.CrossEntropyLoss()(cResult,label2)
#
#             tempRmseLoss += loss1.detach().to(device='cpu').item()
#             tempCrossLoss += loss2.detach().to(device='cpu').item()
#             maxIndex = torch.argmax(cResult.detach().to(device='cpu'),dim=1).view(-1,1)
#             label2 = label2.to(device='cpu').view(-1,1)
#             tempAccuracy += torch.sum(label2==maxIndex)/label2.shape[0]
#
#         tempRmseLoss = tempRmseLoss / len(testLoader)
#         tempCrossLoss = tempCrossLoss / len(testLoader)
#         tempAccuracy = tempAccuracy / len(testLoader)
#
#         testRmseLossEpoch.append(tempRmseLoss)
#         testCrossLossEpoch.append(tempCrossLoss)
#         testAccuracyEpoch.append(tempAccuracy.item())
#
#         print(f'epoch:{e} Test Loss: {tempRmseLoss:.4f}, Test Accuracy:{tempAccuracy.item():.4f}')
#
#     trainRmseLossEpoch = np.array(trainRmseLossEpoch)
#     trainCrossLossEpoch = np.array(trainCrossLossEpoch)
#     trainAccuracyEpoch = np.array(trainAccuracyEpoch)
#
#     testRmseLossEpoch = np.array(testRmseLossEpoch)
#     testCrossLossEpoch = np.array(testCrossLossEpoch)
#     testAccuracyEpoch = np.array(testAccuracyEpoch)
#
#     return  trainRmseLossEpoch, trainCrossLossEpoch, trainAccuracyEpoch, testRmseLossEpoch, testCrossLossEpoch, testAccuracyEpoch
# %%
# class MyDataSet(Dataset):
#     def __init__(self, path):
#         super().__init__()
#         rawData = np.load(path)
#         self.data = rawData['data']
#         self.label1 = rawData['label1']
#         self.label2 = rawData['label2']
#
#     def __getitem__(self, item):
#         data = torch.from_numpy(self.data[item])
#         label1 = torch.from_numpy(self.label1[item])
#         label2 = torch.from_numpy(self.label2[item]).to(dtype=torch.long)
#         return data, label1, label2
#
#     def __len__(self):
#         return self.data.shape[0]


