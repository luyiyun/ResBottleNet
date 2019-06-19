import os
import json
import time

import argparse
import torch.nn as nn


class Config:
    brca_root = "E:/Python/code_genomics/pytorch/Attention/DATA/BRCA"
    brca_cli = os.path.join(brca_root, 'BRCA_clinicalMatrix')
    brca_rna = os.path.join(brca_root, 'HiSeqV2')
    pan_root = "E:/Python/code_genomics/pytorch/Attention/DATA/Pan"
    pan_cli = os.path.join(pan_root, 'PANCAN_clinicalMatrix')
    pan_rna = os.path.join(pan_root, 'HiSeqV2')

    brca_label_mapper = {
        "LumA": 0,
        "LumB": 1,
        "Basal": 2,
        "Normal": 3,
        "Her2": 4
    }

    acts = {
        'leaky_relu': nn.LeakyReLU(),
        'tanh': nn.Tanh()
    }

    def __init__(self,):
        self.parser = argparse.ArgumentParser()

        self.preprocessing_config()  # 预处理的配置
        self.train_config()  # 训练相关配置
        self.net_config()  # attention net相关配置
        self.loss_config()

        self.args = self.parser.parse_args()
        localtime = time.strftime("%Y-%m-%d_%H=%M", time.localtime())
        self.save_dir = self.args.save + '_' + localtime

        # 在初始化配置的同时就创建了保存结果的文件夹，并将配置保存在其中
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.save(os.path.join(self.save_dir, 'config.json'))

        self.act = self.acts[self.args.act]

    def preprocessing_config(self):
        self.parser.add_argument(
            '-d', '--data', default='brca',
            help="使用的数据类型，默认是brca，不同的数据类型决定不同的ML任务"
        )
        self.parser.add_argument(
            '-s', '--save', default='./RESULTS/save',
            help=(
                "结果保存的路径，默认是RESULTS/save，真正创建"
                "的时候会在后面再加上时间"
            )
        )
        self.parser.add_argument(
            '-cv', '--cross_valid', type=int, default=1,
            help=(
                "是否进行cv，以及进行cv的次数，默认是1，代表只进行一次cv split"
                "，此时等价于普通的train-test分割"
            )
        )
        self.parser.add_argument(
            '-ts', '--test_size', type=float, default=0.1,
            help="test size, default 0.1"
        )

    def train_config(self):
        self.parser.add_argument(
            '-bs', '--batch_size', default=32, type=int,
            help='batch size, default 32'
        )
        self.parser.add_argument(
            '-e', '--epoch', default=50, type=int,
            help="epoch, default 50"
        )
        self.parser.add_argument(
            '-lr', '--learning_rate', default=0.01, type=float,
            help="learning rate, default 0.01"
        )
        self.parser.add_argument(
            '-smi', '--standard_metric_index', type=int, default=2,
            help="用于选择最好模型的metric的编号，默认是2，即balanced acc"
        )

    def net_config(self):
        self.parser.add_argument(
            '--net_type', default='atten', choices=['atten', 'mlp', 'resnet'],
            help="使用的网络类型，默认是atten，还可以是mlp或resnet"
        )
        self.parser.add_argument(
            '--hidden_num', default=500, type=int,
            help="self-attention网络使用的隐层节点数，默认是500"
        )
        self.parser.add_argument(
            '--bottle_num', default=50, type=int,
            help="self-attention网络的瓶颈层节点数，默认是50"
        )
        self.parser.add_argument(
            '--block_num', default=4, type=int,
            help="self-attention网络堆叠的attention-linear block的个数，默认是4"
        )
        self.parser.add_argument(
            '--no_res', action='store_false',
            help="当使用此参数时，此参数为false，送入residual参数，即没有残差连接"
        )
        self.parser.add_argument(
            '--act', default='leaky_relu',
            help='bottle neck使用的激活函数，默认leaky_relu'
        )
        self.parser.add_argument(
            '--no_head', action='store_false',
            help="当使用此参数时，此参数为false，送入head_bool参数，即没有head"
        )
        self.parser.add_argument(
            '--no_bottle', action='store_true',
            help="当使用此参数时，此参数为true，送入linear参数，即没有bottle neck"
        )
        self.parser.add_argument(
            '--no_atten', action='store_true',
            help="当使用此参数时，此参数为true，送入no_atten参数，即没有attention"
        )

    def loss_config(self):
        self.parser.add_argument(
            '--loss_type', default='cox', choices=['cox', 'svm'],
            help="loss的类型，为cox或svm，默认是cox"
        )
        self.parser.add_argument(
            '--svm_rankratio', default=1.0, type=float,
            help="当选择svm loss的时候，其rank loss所占的比例，默认是1，必须是0-1"
        )
        self.parser.add_argument(
            '-l2', default=0.0005, type=float,
            help="l2正则化的系数，默认是0.0005"
        )

    def save(self, fn):
        with open(fn, 'w') as f:
            json.dump(self.args.__dict__, f)


config = Config()
