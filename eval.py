# -*- coding: utf-8 -*-
"""
Created on 2023/8/31 18:03 
@Author: Wu Kaixuan
@File  : eval.py 
@Desc  : eval 
"""
import os
from functools import partial
import pandas as pd
import mindspore as ms
from tqdm import tqdm
from src.turbulencenet import TurbulenceNet2
from src.loss import CombinedLoss, R2
from mindspore.train import Metric, MAE
from src.dataset import create_normal_dataset, feature_transforms


def inference(dataset,
              pretrained):
    input_dim = dataset.input_dim
    net = TurbulenceNet2(input_dim=input_dim)
    metrics = {"mae": MAE(), "r2": R2()}
    para_dict = ms.load_checkpoint(pretrained)
    not_load, _ = ms.load_param_into_net(net, para_dict)
    if not_load:
        print(f"Not load pre-trained parameters into net: {not_load}")
    net.set_train(False)
    eval_dataset = dataset.create_dict_iterator()
    for metric in metrics.values():
        metric.clear()
    for batch, data in tqdm(enumerate(eval_dataset), total=dataset.get_dataset_size()):
        output = net(data["node"])
        for metric in metrics.values():
            metric.update(output, data["target"])
    print(f"Eval: " + " ".join([f"{name}: {metric.eval():.6f}" for name, metric in metrics.items()]))
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TurbulenceNet')
    parser.add_argument('--data_root', type=str, default="./data",
                        help='data root directory')
    parser.add_argument('--csv_file', type=str, default="duct_Re2400.csv", help='csv file name: xxx.csv')
    parser.add_argument('--fold_num', type=int, default=10, help='模型数量')
    args_ = parser.parse_args()
    data_root = args_.data_root
    data = pd.read_csv(os.path.join(data_root, args_.csv_file))
    fold_num = args_.fold_num
    # 特征工程  选择特征
    partial_func = partial(feature_transforms,
                           velocity=True,
                           polynomial=True,
                           derivative=True,
                           vorticity=True,
                           gradient=False, )
    # 必须包含这些列
    x = ['U', 'V', 'W', 'U_x', 'U_y', 'U_z', 'V_x', 'V_y', 'V_z', 'W_x', 'W_y', 'W_Z', ]
    y = ['UU', 'UV', 'UW', 'VV', 'VW', 'WW']
    # 检查是否包含这些列, 如果没有则添加
    for col in x:
        if col not in data.columns:
            data[col] = 0
    for col in y:
        if col not in data.columns:
            data[col] = 0
    df_x = data[x]
    df_y = data[y]
    print(f"df_x shape: {df_x.shape}")
    print(f"df_y shape: {df_y.shape}")
    # 保存数据
    df_x.to_csv(os.path.join(data_root, "X.csv"), index=False)
    df_y.to_csv(os.path.join(data_root, "y.csv"), index=False)
    val_data = create_normal_dataset(data_root=data_root,
                                     x_path='X.csv',
                                     y_path='y.csv',
                                     batch_size=800000,
                                     shuffle=True,
                                     normalization="zscore",
                                     split='val',
                                     feature_engineering=partial_func,
                                     )
    # print(val_data.input_dim)
    metrics_value = {"mae": 0, "r2": 0}
    # 10折交叉验证, 取平均
    for i in tqdm(range(fold_num)):
        pretrained = os.path.join("logs/exp46/fold" + str(i + 1), "ms_best.ckpt")
        metrics = inference(val_data, pretrained)
        for name, metric in metrics.items():
            metrics_value[name] += metric.eval()
    for name, metric in metrics_value.items():
        metrics_value[name] = metric / fold_num
        print(f"{name}: {metric / fold_num:.6f}")
    # 保存结果到json文件
    import json
    with open(os.path.join(data_root, "metrics_result.json"), "w") as f:
        json.dump(metrics_value, f)
