# -*- coding: utf-8 -*-
"""
Created on 2023/8/29 11:06 
@Author: Wu Kaixuan
@File  : loss.py 
@Desc  : loss 
"""
import mindspore as ms
import mindspore.nn as nn
from mindspore.nn import L1Loss


class R2(ms.train.Metric):
    '''
    R_squared metric
    '''

    def __init__(self):
        super(R2, self).__init__()
        self.clear()

    def clear(self):
        self.y_pred = None
        self.y_true = None
        self.flag = 0

    def update(self, *inputs):
        if self.flag == 0:
            self.y_pred = inputs[0]
            self.y_true = inputs[1]
            self.flag = 1
        else:
            self.y_pred = ms.ops.concat((self.y_pred, inputs[0]))
            self.y_true = ms.ops.concat((self.y_true, inputs[1]))

    def eval(self):
        residual = ms.ops.sum((self.y_true - self.y_pred) ** 2)
        total = ms.ops.sum((self.y_true - ms.ops.mean(self.y_true)) ** 2)
        r2 = 1 - (residual / total)
        r2 = r2.asnumpy()
        return r2.item()


class ValLoss(ms.train.Metric):
    def __init__(self):
        super(ValLoss, self).__init__()
        self.loss = CombinedLoss()
        self.clear()

    def clear(self):
        self._sum = 0
        self._count = 0

    def update(self, *inputs):
        loss = self.loss(*inputs)
        loss = loss.asnumpy()
        self._sum += loss
        self._count += 1

    def eval(self):
        return self._sum / self._count


class RSquaredLoss(nn.Cell):
    '''
    R_squared loss function
    r_squared  = 1-sum((y_true - y_pred) ** 2) / sum((y_true - mean(y_true)) ** 2)
    越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好
    越接近0，表明模型拟合的越差
    理论上取值范围（-∞，1], 正常取值范围为[0 1]
    '''

    def __init__(self):
        super(RSquaredLoss, self).__init__()

    def construct(self, y_pred, y_true):
        residual = ms.ops.sum((y_true - y_pred) ** 2)
        total = ms.ops.sum((y_true - ms.ops.mean(y_true)) ** 2)
        r2 = 1 - (residual / total)
        return 1 - r2


class CombinedLoss(nn.Cell):
    '''
    Combined loss function
    combined_loss = alpha * MAE + (1 - alpha) * (1 - R_squared)
    '''

    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1loss = L1Loss()

    def construct(self, y_pred, y_true):
        # MAE
        mae = self.l1loss(y_pred, y_true)
        # R-squared
        residual = ms.ops.sum((y_true - y_pred) ** 2)
        total = ms.ops.sum((y_true - ms.ops.mean(y_true)) ** 2)
        r2 = 1 - (residual / total)
        # Combined Loss
        loss = self.alpha * mae + (1 - self.alpha) * (1 - r2)
        return loss, (mae, r2)


class CustomLoss(nn.Cell):
    '''
    Custom loss function
    combined_loss = alpha * MAE + beta * (1 - R_squared) + gamma * physics_loss
    物理约束: 尝试约束模型输出以满足某些物理规则。速度相关矩阵应为半正定。
    Todo: eigenvalues = ms.ops.eigvals(R)  # 注意：需要mindspore 2.0.1
    '''

    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1loss = L1Loss()

    def forward(self, y_pred, y_true):
        # 基础损失: MSE
        mae = self.l1loss(y_pred, y_true)
        # R-squared
        residual = ms.ops.sum((y_true - y_pred) ** 2)
        total = ms.ops.sum((y_true - ms.ops.mean(y_true)) ** 2)
        r2 = 1 - (residual / total)
        # 构建速度协方差矩阵
        physics_loss = 0.0
        # 先将y_pred分解为对应的6个分量
        a, b, c, d, e, f = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3], y_pred[:, 4], y_pred[:, 5]
        # 构建R矩阵，shape为 [batch_size, 3, 3]
        R = ms.ops.stack([
            a, b, c,
            b, d, e,
            c, e, f
        ], axis=-1).reshape(-1, 3, 3)
        # 计算所有矩阵的特征值，shape为 [batch_size, 3]
        # from mindspore.scipy.linalg import eigh
        # eigenvalues = eigh(R)
        eigenvalues = ms.ops.eigvals(R)  # 注意：需要mindspore 2.0.1
        # 计算最小特征值, shape为 [batch_size]
        min_eigenvalues = ms.ops.min(eigenvalues, axis=1).values
        # 物理损失: 如果最小特征值小于零则进行惩罚
        physics_loss = ms.ops.mean(ms.ops.relu(-min_eigenvalues))
        # 总损失: 基础损失和物理损失的加权和
        total_loss = self.alpha * mae + self.beta * (1 - r2) + self.gamma * physics_loss
        return total_loss, (mae, r2)
