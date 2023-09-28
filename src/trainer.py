# -*- coding: utf-8 -*-
"""
Created on 2023/8/31 18:03 
@Author: Wu Kaixuan
@File  : trainer.py
@Desc  : trainer
"""
import os
import random
import numpy as np
import mindspore as ms
from tqdm import tqdm
from mindspore import ops, nn
from mindspore.amp import StaticLossScaler, all_finite
# tensorboardX如果不需要可以注释掉
from tensorboardX import SummaryWriter
from mindspore.communication.management import init, get_rank, get_group_size

from src.dataset import create_dataset,create_normal_dataset

def seed_fixed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    ms.set_seed(seed)
    random.seed(seed)


def init_env(cfg):
    """初始化运行时环境."""
    seed_fixed(cfg.seed)
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)
    # 配置运行模式，支持图模式和PYNATIVE模式
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)
    cfg.device_target = ms.get_context("device_target")
    # 如果是CPU上运行的话，不配置多卡环境
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0
    # 设置运行时使用的卡
    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)

    if cfg.device_num > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)


def r_square(pred, target):
    '''
    越接近1，表明方程的变量对y的解释能力越强，这个模型对数据拟合的也较好
    越接近0，表明模型拟合的越差
    理论上取值范围（-∞，1], 正常取值范围为[0 1]
    :param pred:
    :param target:
    :return:
    '''
    return 1 - (ms.ops.sum((pred - target) ** 2) / ms.ops.sum((target - target.mean()) ** 2))


class Trainer:
    def __init__(self,
                 net,
                 loss,
                 optimizer,
                 train_dataset,
                 eval_dataset,
                 loss_scale=1.0,
                 out_path="./logs/exp1",
                 pretrained=None,
                 tensorboard=False,
                 ):
        '''
        :param net: 网络
        :param loss: 损失函数
        :param optimizer: 优化器
        :param train_dataset: 训练集
        :param eval_dataset: 验证集
        :param loss_scale: 损失放大倍数
        :param out_path: 日志输出路径
        :param pretrained: 预训练模型路径
        :param tensorboard: 是否使用tensorboard
        '''
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        self.net = net
        if pretrained:
            para_dict = ms.load_checkpoint(pretrained)
            not_load, _ = ms.load_param_into_net(self.net, para_dict)
            print(f"Load pre-trained parameters into net: {not_load}")
        self.loss = loss
        self.opt = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_data_size = self.train_dataset.get_dataset_size()  # 获取训练集batch数
        self.weights = self.opt.parameters
        # 注意value_and_grad的第一个参数需要是需要做梯度求导的图，一般包含网络和loss。这里可以是一个函数，也可以是Cell
        self.value_and_grad = ms.value_and_grad(self.forward_fn, None, weights=self.weights, has_aux=True)
        # 分布式场景使用
        self.grad_reducer = self.get_grad_reducer()
        self.loss_scale = StaticLossScaler(loss_scale)
        self.run_eval = eval_dataset is not None
        if self.run_eval:
            self.eval_dataset = eval_dataset
            self.eval_data_size = self.eval_dataset.get_dataset_size()  # 获取验证集batch数
            self.best_loss = 0
            self.best_mae = 0
            self.best_r2 = 0
        self.tensorboard = tensorboard

    def get_grad_reducer(self):
        grad_reducer = ops.identity
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        # 判断是否是分布式场景，分布式场景的设置参考上面通用运行环境设置
        reducer_flag = (parallel_mode != ms.ParallelMode.STAND_ALONE)
        if reducer_flag:
            grad_reducer = nn.DistributedGradReducer(self.weights)
        return grad_reducer

    def forward_fn(self, inputs, labels):
        """正向网络构建，注意第一个输出必须是最后需要求梯度的那个输出"""
        logits = self.net(inputs)
        loss, mae_r2 = self.loss(logits, labels)
        loss = self.loss_scale.scale(loss)
        return loss, mae_r2

    @ms.jit  # jit加速，需要满足图模式构建的要求，否则会报错
    def train_single(self, inputs, labels):
        (loss, mae_r2), grads = self.value_and_grad(inputs, labels)
        loss = self.loss_scale.unscale(loss)
        grads = self.loss_scale.unscale(grads)
        grads = self.grad_reducer(grads)
        state = all_finite(grads)
        if state:
            self.opt(grads)
        return loss, mae_r2

    def train(self, epochs):
        if self.tensorboard:
            writer = SummaryWriter(self.out_path)
        else:
            writer = None
        train_dataset = self.train_dataset.create_dict_iterator()
        self.net.set_train(True)
        for epoch in tqdm(range(epochs)):
            train_loss = 0
            train_r2 = 0
            train_mae = 0
            # 训练一个epoch
            for batch, data in tqdm(enumerate(train_dataset), total=self.train_data_size):
                loss, mae_r2 = self.train_single(data["node"], data["target"])
                train_loss += loss
                train_r2 = mae_r2[1]
                train_mae += mae_r2[0]
            print(f"epoch {epoch} train: loss: {train_loss.asnumpy() / (batch + 1) :.6f} "
                  f"mae: {train_mae.asnumpy() / (batch + 1) :.6f} "
                  f"r2: {train_r2.asnumpy() :.6f}")
            if self.tensorboard:
                writer.add_scalar("train/train_loss", train_loss.asnumpy() / (batch + 1), epoch)
                writer.add_scalar("train/train_mae", train_mae.asnumpy() / (batch + 1), epoch)
                writer.add_scalar("train/train_r2", train_r2.asnumpy(), epoch)
                writer.add_scalar("train/lr", self.opt.get_lr().asnumpy(), epoch)
            ms.save_checkpoint(self.net, os.path.join(self.out_path, "last.ckpt"))
            # 推理并保存最好的那个checkpoint
            if self.run_eval:
                eval_dataset = self.eval_dataset.create_dict_iterator()
                self.net.set_train(False)
                val_loss = 0
                val_mae = 0
                val_r2 = 0
                for batch, data in tqdm(enumerate(eval_dataset), total=self.eval_data_size):
                    loss, mae_r2 = self.forward_fn(data["node"], data["target"])
                    val_r2 = mae_r2[1]
                    val_loss += loss
                    val_mae += mae_r2[0]
                print(f"epoch {epoch} val: loss: {val_loss.asnumpy() / (batch + 1) :.6f} "
                      f"mae: {val_mae.asnumpy()/(batch+1) :.6f} "
                      f"r2: {val_r2.asnumpy() :.6f}")
                if self.tensorboard:
                    writer.add_scalar("val/loss", val_loss.asnumpy() / (batch + 1), epoch)
                    writer.add_scalar("val/mae", val_loss.asnumpy() / (batch + 1), epoch)
                    writer.add_scalar("val/r2", val_r2.asnumpy(), epoch)
                if val_r2 >= self.best_r2:
                    # 保存最好的那个checkpoint
                    self.best_r2 = val_r2
                    ms.save_checkpoint(self.net, os.path.join(self.out_path, "best.ckpt"))
                    print(f"Updata best R2: {val_r2}")
                self.net.set_train(True)
