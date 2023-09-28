# -*- coding: utf-8 -*-
"""
Created on 2023/8/29 10:58 
@Author: Wu Kaixuan
@File  : turbulencenet.py 
@Desc  : turbulencenet 
"""
import mindspore.nn as nn
from mindspore.common.initializer import HeUniform, initializer, Uniform


class TurbulenceNet(nn.Cell):
    def __init__(self, input_dim=12, output_dim=6):
        super(TurbulenceNet, self).__init__()

        self.layer1 = nn.SequentialCell(
            nn.Dense(input_dim, 64),
            nn.ReLU(),
        )
        self.layer2 = nn.SequentialCell(
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dense(32, 16),
            nn.ReLU(),
            nn.Dense(16, 16),
            nn.ReLU(),
            nn.Dense(16, output_dim),
        )
        self.weight_init()

    def weight_init(self):
        for name, param in self.layer1.parameters_and_names():
            if "weight" in name:
                param.set_data(initializer(HeUniform(), param.shape, param.dtype))
            if "bias" in name:
                param.set_data(initializer(Uniform(), param.shape, param.dtype))
        for name, param in self.layer2.parameters_and_names():
            if "weight" in name:
                param.set_data(initializer(HeUniform(), param.shape, param.dtype))
            if "bias" in name:
                param.set_data(initializer(Uniform(), param.shape, param.dtype))

    def construct(self, x):
        x = self.layer1(x)
        return self.layer2(x)


class TurbulenceNet2(nn.Cell):
    def __init__(self, input_dim=12, output_dim=6):
        super(TurbulenceNet2, self).__init__()

        self.layer1 = nn.SequentialCell(
            nn.Dense(input_dim, 128, ),
            nn.ReLU(),
        )
        self.layer2 = nn.SequentialCell(
            nn.Dense(128, 64, ),
            nn.ReLU(),
            nn.Dense(64, 64, ),
            nn.ReLU(),
            nn.Dense(64, 32, ),
            nn.ReLU(),
            nn.Dense(32, output_dim),
        )
        self.weight_init()

    def weight_init(self):
        for name, param in self.layer1.parameters_and_names():
            if "weight" in name:
                param.set_data(initializer(HeUniform(), param.shape, param.dtype))
            if "bias" in name:
                param.set_data(initializer(Uniform(), param.shape, param.dtype))
        for name, param in self.layer2.parameters_and_names():
            if "weight" in name:
                param.set_data(initializer(HeUniform(), param.shape, param.dtype))
            if "bias" in name:
                param.set_data(initializer(Uniform(), param.shape, param.dtype))

    def construct(self, x):
        x = self.layer1(x)
        return self.layer2(x)
