# -*- coding: utf-8 -*-
"""
Created on 2023/8/29 11:13
@Author: Wu Kaixuan
@File  : dataset.py
@Desc  : dataset
"""
import os
from typing import Literal
import joblib
import mindspore as ms
import mindspore.dataset as ds
import pandas as pd
import numpy
import mindspore.dataset.transforms as transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def feature_transforms(data,
                       velocity=True,
                       polynomial=True,
                       derivative=True,
                       vorticity=True,
                       gradient=False,
                       ):
    '''
    特征工程
    :param data: 输入数据 默认True
    :param velocity: 速度交互 默认True
    :param polynomial: 多项式交互 默认True
    :param derivative: 导数交互 默认True
    :param vorticity: 涡量交互  默认True
    :param gradient: 梯度交互   默认False
    '''
    # 速度交互
    if velocity:
        data['U_V'] = data['U'] * data['V']
        data['U_W'] = data['U'] * data['W']
        data['V_W'] = data['V'] * data['W']
    # 多项式交互
    if polynomial:
        data['U2'] = data['U'] ** 2
        data['V2'] = data['V'] ** 2
        data['W2'] = data['W'] ** 2
    # 导数交互
    if derivative:
        data['U_xV_y'] = data['U_x'] * data['V_y']
        data['U_xW_z'] = data['U_x'] * data['W_Z']
        data['V_yW_z'] = data['V_y'] * data['W_Z']
    # 梯度（慎用）只有输入数据是连续的时候才能使用
    if gradient:
        # 用于时间序列一阶差分
        data['U_grad'] = data['U'].diff()
        data['V_grad'] = data['V'].diff()
        data['W_grad'] = data['W'].diff()
    # 涡量
    if vorticity:
        data['Omega_x'] = data['W_y'] - data['V_z']
        data['Omega_y'] = data['U_z'] - data['W_x']
        data['Omega_z'] = data['V_x'] - data['U_y']
    return data


class TurbulenceDataset:
    '''
    自定义数据集类
    '''

    def __init__(self,
                 data_root: str,
                 x_path: str,
                 y_path: str,
                 split: Literal['train', 'test', 'val'] = "val",
                 normalization=False, eps=1e-8, ):
        '''
        :param data_root: 文件根目录
        :param x: 训练输入数据文件
        :param y: 训练输出数据文件
        :param split: 数据划分
        :param normalization: 归一化
        :param eps: 防止数值溢出
        '''
        self.x = pd.read_csv(os.path.join(data_root, x_path))
        self.y = pd.read_csv(os.path.join(data_root, y_path))
        # self.x = self.x[:10]
        # self.y = self.y[:10]
        print(f"Load {split} data successfully! data size:{self.x.shape[0]}")
        assert self.x.shape[0] == self.y.shape[0]
        if split == 'train':
            self.mean = self.x.mean()
            self.std = self.x.std()
            # 保存均值和方差到文件
            numpy.savetxt(os.path.join(data_root, "mean.txt"), self.mean)
            numpy.savetxt(os.path.join(data_root, "std.txt"), self.std)
        else:
            self.mean = numpy.loadtxt(os.path.join(data_root, "mean.txt"))
            self.std = numpy.loadtxt(os.path.join(data_root, "std.txt"))
        if normalization:
            self.x = (self.x - self.mean) / (self.std + eps)
        self.x = self.x.to_numpy()
        self.y = self.y.to_numpy()

    def __getitem__(self, item):
        node = self.x[item]
        target = self.y[item]
        return node, target

    def __len__(self):
        return self.x.shape[0]


class ToTalNormalDataset:
    def __init__(self,
                 data_root,
                 x_path,
                 y_path,
                 normalization="zscore",
                 split="train",
                 feature_engineering=feature_transforms):
        self.x = pd.read_csv(os.path.join(data_root, x_path))
        self.y = pd.read_csv(os.path.join(data_root, y_path))
        self.x.fillna(0, inplace=True)
        self.y.fillna(0, inplace=True)
        print(f"Load data from {data_root}, x:{self.x.shape}, y:{self.y.shape}")
        print(f"Normalization: {normalization}")
        assert self.x.shape[0] == self.y.shape[0]
        if feature_engineering:
            self.x = feature_engineering(self.x)
            print(f"Feature engineering: {self.x.shape[1]}")
        setattr(self, "input_dim", self.x.shape[1])
        self.x = self.x.to_numpy(dtype=numpy.float32)
        self.y = self.y.to_numpy(dtype=numpy.float32)
        # 归一化一定要注意，要用训练集的均值和方差
        if normalization == "minmax":
            if split == "train":
                if os.path.exists(os.path.join(data_root, "minmax.pkl")):
                    print(f"Load minmax from {os.path.join(data_root, 'minmax.pkl')}")
                    self.scaler = joblib.load(os.path.join(data_root, "minmax.pkl"))
                    self.x = self.scaler.transform(self.x)
                else:
                    self.scaler = MinMaxScaler()
                    self.x = self.scaler.fit_transform(self.x)
                    joblib.dump(self.scaler, os.path.join(data_root, "minmax.pkl"))
                    print(f"Save minmax to {os.path.join(data_root, 'minmax.pkl')}")
            elif split == "val":
                if os.path.exists(os.path.join(data_root, "minmax.pkl")):
                    print(f"Load minmax from {os.path.join(data_root, 'minmax.pkl')}")
                    self.scaler = joblib.load(os.path.join(data_root, "minmax.pkl"))
                    self.x = self.scaler.transform(self.x)
                else:
                    raise Exception("minmax.pkl not found!")
        elif normalization == "zscore":
            if split == "train":
                if os.path.exists(os.path.join(data_root, "zscore.pkl")):
                    try:
                        print(f"Load zscore from {os.path.join(data_root, 'zscore.pkl')}.....")
                        self.scaler = joblib.load(os.path.join(data_root, "zscore.pkl"))
                        self.x = self.scaler.transform(self.x)
                        print(f"Load zscore from {os.path.join(data_root, 'zscore.pkl')} success!")
                    except Exception as e:
                        print(f"Load zscore from {os.path.join(data_root, 'zscore.pkl')} failed, because {e}")
                        print(f"Recompute zscore.....")
                        self.scaler = StandardScaler()
                        self.x = self.scaler.fit_transform(self.x)
                        joblib.dump(self.scaler, os.path.join(data_root, "zscore.pkl"))
                        print(f"Save zscore to {os.path.join(data_root, 'zscore.pkl')}")
                else:
                    print(f"Recompute zscore.....")
                    self.scaler = StandardScaler()
                    self.x = self.scaler.fit_transform(self.x)
                    joblib.dump(self.scaler, os.path.join(data_root, "zscore.pkl"))
                    print(f"Save zscore to {os.path.join(data_root, 'zscore.pkl')}")
            elif split == "val":
                if os.path.exists(os.path.join(data_root, "zscore.pkl")):
                    print(f"Load zscore from {os.path.join(data_root, 'zscore.pkl')}.....")
                    self.scaler = joblib.load(os.path.join(data_root, "zscore.pkl"))
                    self.x = self.scaler.transform(self.x)
                    print(f"Load zscore from {os.path.join(data_root, 'zscore.pkl')} success!")
                else:
                    raise Exception("zscore.pkl not found!")

        else:
            self.x = self.x
            self.scaler = None

    def __getitem__(self, item):
        node = self.x[item]
        target = self.y[item]
        return node, target

    def __len__(self):
        return self.x.shape[0]


def create_dataset(data_root,
                   x,
                   y,
                   batch_size=1,
                   split="train",
                   normalization=True, ):
    '''
    :param data_root: 数据根目录
    :param batch_size: 批大小
    :param x: 输入数据文件
    :param y: 输出数据文件
    :param split: 数据划分
    :param normalization: 是否归一化
    :return: data
    '''
    data = ds.GeneratorDataset(
        TurbulenceDataset(data_root=data_root,
                          x_path=x,
                          y_path=y,
                          split=split,
                          normalization=normalization),
        python_multiprocessing=True,
        column_names=['node', 'target'],
        shuffle=True,
        num_parallel_workers=1,
    )
    data = data.map(operations=transforms.TypeCast(ms.float32))
    data = data.batch(batch_size)
    return data


def create_normal_dataset(data_root,
                          x_path,
                          y_path,
                          batch_size=1,
                          normalization="zscore",
                          split="train",
                          feature_engineering=feature_transforms,
                          shuffle=True, ):
    '''
    :param data_root: 数据根目录
    :param batch_size: 批大小
    :param x_path: 输入数据文件
    :param y_path: 输出数据文件
    :param normalization: 归一化[None, 'minmax', 'zscore']
    :param split: 数据划分
    :param feature_engineering: 特征工程
    :param shuffle: 是否打乱
    :return: data
    '''
    custom_dataset = ToTalNormalDataset(data_root=data_root,
                                        x_path=x_path,
                                        y_path=y_path,
                                        split=split,
                                        normalization=normalization,
                                        feature_engineering=feature_engineering)

    data = ds.GeneratorDataset(
        custom_dataset,
        python_multiprocessing=True,
        column_names=['node', 'target'],
        shuffle=shuffle,
        num_parallel_workers=1,
    )
    data = data.map(operations=transforms.TypeCast(ms.float32))
    data = data.batch(batch_size)
    setattr(data, "input_dim", custom_dataset.input_dim)
    return data


if __name__ == '__main__':
    # datas = create_dataset(data_root='F:\TurbulenceModeling\TurbulenceMS\data\dropna',
    #                        x='X_train.csv',
    #                        y='y_train.csv',
    #                        split='train',
    #                        batch_size=256,
    #                        normalization=True)
    datas = create_normal_dataset(data_root=r'F:\TurbulenceModeling\TurbulenceMS\data\fillna',
                                  x_path='X_train.csv',
                                  y_path='y_train.csv',
                                  batch_size=256,
                                  split="train",
                                  normalization="zscore")
    print(datas.get_dataset_size())
