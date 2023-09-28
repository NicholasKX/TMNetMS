# -*- coding: utf-8 -*-
"""
Created on 2023/8/30 16:36 
@Author: Wu Kaixuan
@File  : exp2.py 
@Desc  : exp2 
"""
import os
from functools import partial

import mindspore as ms
from mindspore import nn
from src.turbulencenet import TurbulenceNet2
from src.loss import CombinedLoss
from mindspore.nn import Adam
from src.trainer import Trainer, seed_fixed
from src.dataset import create_normal_dataset, feature_transforms

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")


def main(args):
    # 固定随机种子
    seed_fixed()
    if args['pretrained_path'] == "None":
        pretrained_path = None
    else:
        pretrained_path = args['pretrained_path']
    partial_func = partial(feature_transforms,
                           velocity=True,
                           polynomial=True,
                           derivative=True,
                           vorticity=True,
                           gradient=False, )
    # 数据集加载
    train_data = create_normal_dataset(data_root=args['data_dir'],
                                       x_path=args['train_input_path'],
                                       y_path=args['train_output_path'],
                                       feature_engineering=partial_func,
                                       batch_size=args['batch_size'], )
    step_size_train = train_data.get_dataset_size()
    if args['val_input_path'] and args['val_output_path']:
        val_data = create_normal_dataset(data_root=args['data_dir'],
                                         x_path=args['val_input_path'],
                                         y_path=args['val_output_path'],
                                         feature_engineering=partial_func,
                                         split='val',
                                         batch_size=args['batch_size'],
                                         )
        step_size_val = val_data.get_dataset_size()
    else:
        val_data = None
        step_size_val = 0
    # 模型定义
    model = TurbulenceNet2(train_data.input_dim)
    # 损失函数定义
    loss_fn = CombinedLoss()
    # 优化器定义和学习率定义
    lr = nn.cosine_decay_lr(min_lr=1e-7, max_lr=1e-3,
                            decay_epoch=args["epochs"],
                            total_step=step_size_train * args["epochs"],
                            step_per_epoch=step_size_train)
    optimizer = Adam(model.trainable_params(), learning_rate=lr, weight_decay=0.0001)
    # 训练器定义
    trainer = Trainer(net=model,
                      loss=loss_fn,
                      optimizer=optimizer,
                      train_dataset=train_data,
                      eval_dataset=val_data,
                      out_path=os.path.join(args["logs_dir"], args["exp_name"]),
                      pretrained=pretrained_path,
                      tensorboard=args["tensorboard"],
                      )
    trainer.train(args["epochs"])


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='TurbulenceNet')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: GPU)')
    parser.add_argument('--config', type=str, default="config/turbulencenet_config.yaml", help='config file')

    args_ = parser.parse_args()
    # 读取yaml文件
    with open(args_.config, 'r') as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(configs)
    main(configs)
