# -*- coding: utf-8 -*-
"""
Created on 2023/8/30 14:54 
@Author: Wu Kaixuan
@File  : weight_transfer.py 
@Desc  : weight_transfer 
"""

import json
import os.path
import torch
import mindspore as ms
import numpy as np
from tqdm import tqdm

from src.turbulencenet import TurbulenceNet2


# 通过PyTorch参数文件，打印PyTorch的参数文件里所有参数的参数名和shape，返回参数字典
def pytorch_params(pth_file):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    for name, params in par_dict.items():
        print(f"{name} : {params.shape}")
        pt_params[name] = params.detach().numpy()
    return pt_params


# 通过MindSpore的Cell，打印Cell里所有参数的参数名和shape，返回参数字典
def mindspore_params(network):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        print(f"{name} : {value.shape}")
        ms_params[name] = value
    return ms_params


def param_convert(ms_params, pt_params, ckpt_path):
    # 参数名映射字典
    print(ms_params.keys())
    print(pt_params.keys())
    new_params_list = []
    for ms_param in ms_params.keys():
        if ms_param in pt_params.keys():
            pt_param = pt_params[ms_param]
            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param.shape == ms_params[ms_param].shape:
                ms_value = pt_param
                new_params_list.append({"name": ms_param, "data": ms.Tensor(ms_value)})
            else:
                print(ms_param, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)


if __name__ == '__main__':
    # model_weight = "logs/exp46/fold1/"
    # pytorch格式的参数转换成MindSpore格式的参数
    for i in tqdm(range(1, 11)):
        model_weight = "logs/exp46/fold{}/".format(i)
        pt_params = pytorch_params(os.path.join(model_weight, "best.pth"))
        model = TurbulenceNet2(input_dim=24)
        # # para_dict = ms.load_checkpoint("checkpoint/ms_encoder.ckpt")
        # # not_load, _ = ms.load_param_into_net(encoder, para_dict)
        ms_params = mindspore_params(model)
        param_convert(ms_params, pt_params, os.path.join(model_weight, "ms_best.ckpt"))
