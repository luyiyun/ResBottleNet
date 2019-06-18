import os
import copy

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import progressbar as pb
import pandas as pd

from datasets import RnaData
import transfers as tf
from net import SelfAttentionNet, MLP, ResidualNet, NegativeLogLikelihood
import metrics as mm
from config import config


class NoneScheduler:
    def __init__(self):
        pass

    def step(self):
        pass


def train(
    model, criterion, optimizer, dataloaders, scheduler=NoneScheduler(),
    epoch=100, device=torch.device('cuda:0'), l2=0.0,
    metrics=(mm.Loss(), mm.Accuracy()), standard_metric_index=1,
    clip_grad=False
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0
    best_metric_name = metrics[standard_metric_index].__class__.__name__ + \
        '_valid'
    history = {
        m.__class__.__name__+p: []
        for p in ['_train', '_valid']
        for m in metrics
    }

    for e in range(epoch):
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()
                iterator = pb.progressbar(dataloaders[phase], prefix='Train: ')
            else:
                model.eval()
                iterator = dataloaders[phase]
            for m in metrics:
                m.reset()
            for batch_x, batch_y in iterator:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    logit = model(batch_x)
                    loss = criterion(logit, batch_y)
                    # 只给weight加l2正则化
                    if l2 > 0.0:
                        for p_n, p_v in model.named_parameters():
                            if p_n == 'weight':
                                loss += l2 * p_v.norm()
                    if phase == 'train':
                        loss.backward()
                        if clip_grad:
                            nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1)
                        optimizer.step()
                with torch.no_grad():
                    for m in metrics:
                        if isinstance(m, mm.Loss):
                            m.add(loss.cpu().item(), batch_x.size(0))
                        else:
                            m.add(logit, batch_y)

            for m in metrics:
                history[m.__class__.__name__+'_'+phase].append(m.value())
            print(
                "Epoch: %d, Phase:%s, " % (e, phase) +
                ", ".join([
                    '%s: %.4f' % (
                        m.__class__.__name__,
                        history[m.__class__.__name__+'_'+phase][-1]
                    ) for m in metrics
                ])
            )

            if phase == 'valid':
                epoch_metric = history[best_metric_name][-1]
                if epoch_metric > best_metric:
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())

    print("Best metric: %.4f" % best_metric)
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate(
    net, criterion, test_dataloader, metrics, device=torch.device('cuda:0')
):
    history = {}
    for m in metrics:
        m.reset()
    iterator = pb.progressbar(test_dataloader, prefix='Test: ')
    for batch_x, batch_y in iterator:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            logit = net(batch_x)
            loss = criterion(logit, batch_y)
            for m in metrics:
                if isinstance(m, mm.Loss):
                    m.add(loss.cpu().item(), batch_x.size(0))
                else:
                    m.add(logit, batch_y)
    for m in metrics:
        history[m.__class__.__name__] = m.value()
    print(
        "Test results: " +
        ", ".join([
            '%s: %.4f' % (m.__class__.__name__, history[m.__class__.__name__])
            for m in metrics
        ])
    )
    return history


def main():

    # ----- 根据data来读取不同的数据和不同的loss、metrics -----
    if config.args.data == 'brca':
        rna = RnaData.predicted_data(
            config.brca_cli, config.brca_rna,
            {'PAM50Call_RNAseq': 'pam50'}
        )
        rna.transform(tf.LabelMapper(config.brca_label_mapper))
        out_shape = len(config.brca_label_mapper)
        criterion = nn.CrossEntropyLoss()
        scorings = (
            mm.Loss(), mm.Accuracy(), mm.BalancedAccuracy(),
            mm.F1Score(average='macro'), mm.Precision(average='macro'),
            mm.Recall(average='macro'), mm.ROCAUC(average='macro')
        )
    elif config.args.data == 'survival':
        if os.path.exists('./DATA/temp_pan.pth'):
            rna = RnaData.load('./DATA/temp_pan.pth')
        else:
            rna = RnaData.survival_data(
                config.pan_cli, config.pan_rna, '_OS_IND', '_OS')
        out_shape = 1
        criterion = NegativeLogLikelihood()
        scorings = (mm.Loss(), mm.CIndex())
    rna.transform(tf.ZeroFilterCol(0.8))
    rna.transform(tf.MeanFilterCol(1))
    rna.transform(tf.StdFilterCol(0.5))
    norm = tf.Normalization()
    rna.transform(norm)

    # ----- 构建网络和优化器 -----
    inpt_shape = rna.X.shape[1]
    if config.args.net_type == 'mlp':
        net = MLP(
            inpt_shape, out_shape, config.args.hidden_num,
            config.args.block_num
        ).cuda()
    elif config.args.net_type == 'atten':
        net = SelfAttentionNet(
            inpt_shape, out_shape, config.args.hidden_num,
            config.args.bottle_num, config.args.block_num,
            config.args.no_res, config.act, config.args.no_head,
            config.args.no_bottle, config.args.no_atten
        ).cuda()
    elif config.args.net_type == 'resnet':
        net = ResidualNet(
            inpt_shape, out_shape, config.args.hidden_num,
            config.args.bottle_num, config.args.block_num
        ).cuda()
    optimizer = optim.Adamax(net.parameters(), lr=config.args.learning_rate)

    # ----- 训练网络，cross validation -----
    split_iterator = rna.split_cv(
        config.args.test_size, config.args.cross_valid)
    train_hists = []
    test_hists = []
    for split_index, (train_rna, test_rna) in enumerate(split_iterator):
        print(
            '##### save: %s, split: %d #####' %
            (config.args.save, split_index)
        )
        #  从train中再分出一部分用作验证集，决定停止
        train_rna, valid_rna = train_rna.split(0.1)
        dats = {
            'train': train_rna.to_torchdat(),
            'valid': valid_rna.to_torchdat(),
        }
        dataloaders = {
            k: data.DataLoader(v, batch_size=config.args.batch_size)
            for k, v in dats.items()
        }
        test_dataloader = data.DataLoader(
            test_rna.to_torchdat(), batch_size=config.args.batch_size
        )
        # 网络训练前都进行一次参数重置，避免之前的训练的影响
        net.reset_parameters()
        # train
        net, hist = train(
            net, criterion, optimizer, dataloaders, epoch=config.args.epoch,
            metrics=scorings, l2=config.args.l2,
            standard_metric_index=config.args.standard_metric_index
        )
        # test
        test_res = evaluate(net, criterion, test_dataloader, metrics=scorings)
        # 将多次训练的结果保存到一个df中
        hist = pd.DataFrame(hist)
        hist['split_index'] = split_index
        train_hists.append(hist)
        # 保存多次test的结果
        test_res['split_index'] = split_index
        test_hists.append(test_res)
        # 每个split训练的模型保存为一个文件
        torch.save(
            net.state_dict(),
            os.path.join(config.save_dir, 'model%d.pth' % split_index)
        )
    # 保存train的结果
    train_hists = pd.concat(train_hists)
    train_hists.to_csv(os.path.join(config.save_dir, 'train.csv'))
    # 保存test的结果
    test_hists = pd.DataFrame(test_hists)
    test_hists.to_csv(os.path.join(config.save_dir, 'test.csv'))


if __name__ == "__main__":
    main()
