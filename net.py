import torch
import torch.nn as nn
import torch.nn.functional as F


''' 这一部分主要是关于SelfAttentionNet的部分 '''


def bottle_linear(
    in_f, out_f, bottle_f, act=nn.LeakyReLU(), linear=False,
    dropout=0.0
):
    '''
    得到一个bottle neck layer, linear参数如果是True这是普通的linear层，
    此主要用于进行比较；
    '''
    if linear:
        return nn.Linear(in_f, out_f)
    return nn.Sequential(
        nn.Linear(in_f, bottle_f), act, nn.Dropout(dropout, inplace=True),
        nn.Linear(bottle_f, out_f)
    )


class AttentionBlock(nn.Module):
    '''
    一个自注意力机制实现，根据输入使用bottle neck分别计算query和key，然后使用
    query和key计算一个inpt_shape x inpt_shape的注意力分数矩阵，其每一列的和是1，
    表示其中每个元素对其他元素的影响。然后将此矩阵矩阵乘输入，然后再加上输入（
    一个residual连接），最后跟一个bn。
    '''
    def __init__(
        self, input_shape, bottle_unit, bottle_act=nn.LeakyReLU(),
        residual=True, linear=False, dropout=0.0
    ):
        super(AttentionBlock, self).__init__()
        self.residual = residual
        self.query = bottle_linear(
            input_shape, input_shape, bottle_unit, bottle_act, linear, dropout)
        self.key = bottle_linear(
            input_shape, input_shape, bottle_unit, bottle_act, linear, dropout)
        self.bn = nn.BatchNorm1d(input_shape)

    def forward(self, x):
        identity = x
        q = self.query(x)
        k = self.key(x)
        self.s = self._attention_score(k, q)
        out = torch.bmm(self.s, x.unsqueeze(2)).squeeze(2)
        if self.residual:
            out += identity
        return self.bn(out)

    @staticmethod
    def _attention_score(key, query):
        key = key.unsqueeze(2)
        query = query.unsqueeze(1)
        return F.softmax(torch.bmm(key, query), dim=-1)

    @property
    def attention_score(self):
        return self.s.detach()


class AttenLinearBlock(nn.Module):
    '''
    attention-linear模块，即在上面attention模块之后跟一个activation、bottle neck
    、bn、activation组成的一个整体的模块，并在最后的activation前加一个residual
    连接，网络一般使用此结构堆叠形成。
    '''
    def __init__(
        self, in_num, bottle_num, bottle_act=nn.LeakyReLU(), residual=True,
        linear=False, no_atten=False, dropout=0.0
    ):
        super(AttenLinearBlock, self).__init__()
        self.residual = residual
        if no_atten:
            self.attention = bottle_linear(
                in_num, in_num, bottle_num, bottle_act, linear, dropout)
        else:
            self.attention = AttentionBlock(
                in_num, bottle_num, bottle_act, residual, linear, dropout)
        self.linear = bottle_linear(
            in_num, in_num, bottle_num, bottle_act, linear, dropout)
        self.bn = nn.BatchNorm1d(in_num)

    def forward(self, x):
        identity = x
        x = self.bn(self.linear(self.attention(x)))
        if self.residual:
            x += identity
        return F.leaky_relu_(x)

    @property
    def attention_score(self):
        return self.attention.attention_score


class SelfAttentionNet(nn.Module):
    ''' 使用attention-linear模块堆叠形成的网络 '''
    def __init__(
        self, input_shape, out_shape, hidden_num=500, bottle_num=50,
        block_num=3, residual=True, act=nn.LeakyReLU(), head_bool=True,
        linear=False, no_atten=False, dropout=0.0
    ):
        '''
        input_shape,out_shape:输入和输出的维度；
        hidden_num:隐层的大小，每个隐层的大小都是一样的，隐层与隐层之间是
            attention-linear block；
        bottle_num:bottle neck layer中bottle的大小；
        block_num:共叠加了几个attention-linear block；
        act:bottle neck中使用的非线性连接；
        head_bool:是否在前面加入一个bottle neck layer来降维；
        以下三个参数主要在进行比较的时候使用：：：
        residual:是否在attention-linear中加入残差连接；
        linear:如果是True则所有的bottle neck换成普通的linear层；
        no_atten:如果是True则将所有的自注意力layer替换为bottle necklayer；
        '''
        super(SelfAttentionNet, self).__init__()
        if head_bool:
            models = [nn.Linear(input_shape, hidden_num), act]
        else:
            models = []
        models += [
            AttenLinearBlock(
                hidden_num, bottle_num, act, residual, linear, no_atten,
                dropout
            ) for _ in range(block_num)
        ]
        models.append(nn.Linear(hidden_num, out_shape))
        self.models = nn.Sequential(*models)

    def forward(self, x):
        return self.models(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    @property
    def attention_scores(self):
        scores = []
        for m in self.models.children():
            if isinstance(m, AttenLinearBlock):
                scores.append(m.attention_score)
        return scores


''' 单纯的MLP '''


class MLP(nn.Module):
    def __init__(self, input_shape, out_shape, hidden_num=500, block_num=3):
        super(MLP, self).__init__()
        models = [
            nn.Linear(input_shape, hidden_num),
            nn.LeakyReLU(),
        ]
        models += [
            nn.Linear(hidden_num, hidden_num)
            for _ in range(block_num)
        ]
        models.append(nn.Linear(hidden_num, out_shape))
        self.models = nn.Sequential(*models)

    def forward(self, x):
        return self.models(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


''' 这里主要描述了一个只有residual和bottleneck的网络 '''


class ResidualLinearBlock(nn.Module):
    def __init__(self, in_num, bottle_num):
        super(ResidualLinearBlock, self).__init__()
        self.linear1 = bottle_linear(in_num, in_num, bottle_num)
        self.linear2 = bottle_linear(in_num, in_num, bottle_num)
        self.bn = nn.BatchNorm1d(in_num)

    def forward(self, x):
        identity = x
        x = self.bn(self.linear2(self.linear1(x)))
        return F.leaky_relu(identity + x)


class ResidualNet(nn.Module):
    def __init__(
        self, input_shape, out_shape, hidden_num=500,
        bottle_num=100, block_num=3
    ):
        super(ResidualNet, self).__init__()
        models = [
            nn.Linear(input_shape, hidden_num),
            nn.LeakyReLU(),
        ]
        models += [
            ResidualLinearBlock(hidden_num, bottle_num)
            for _ in range(block_num)
        ]
        self.models = nn.Sequential(*models)
        self.end = nn.Linear(hidden_num, out_shape)

    def forward(self, x):
        return self.end(self.models(x))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


''' 预测生存使用的loss '''


class NegativeLogLikelihood(nn.Module):
    def __init__(self, reduction='mean'):
        super(NegativeLogLikelihood, self).__init__()
        self.reduction = reduction

    def forward(self, logit, status_time):
        status, time = status_time[:, 0], status_time[:, 1]
        logit = logit.squeeze()
        index = time.argsort(descending=True)
        logit, status = logit[index], status[index]
        log_risk_delta = logit - logit.exp().cumsum(0).log()
        censored_risk = log_risk_delta * status.float()
        # censored_risk = log_risk_delta[status.byte()]
        if self.reduction == 'sum':
            return -censored_risk.sum()
        return -censored_risk.mean()


class SvmLoss(nn.Module):
    def __init__(self, reduction='mean', rank_ratio=1.0):
        super(SvmLoss, self).__init__()
        if reduction == 'mean':
            self.agg_func = torch.mean
        elif reduction == 'sum':
            self.agg_func = torch.sum
        self.rank_ratio = rank_ratio

    def forward(self, logit, status_time):
        status, time = status_time[:, 0].float(), status_time[:, 1].float()
        logit = -logit.squeeze()  # 加一个-使得我们得到得总是risk
        # rank loss
        low, high = self._comparable_pairs(status, time)
        low_logits, high_logits = logit[low], logit[high]
        rank_loss = self.agg_func(
            (1 - high_logits + low_logits).clamp(min=0) ** 2)
        # regress loss
        uncensor_part = time - logit
        censor_part = uncensor_part.clamp(min=0)
        reg_loss = self.agg_func(
            ((1 - status) * censor_part + status * uncensor_part) ** 2)
        # 总loss
        loss = self.rank_ratio * rank_loss + (1 - self.rank_ratio) * reg_loss
        return loss

    @staticmethod
    def _comparable_pairs(status, time):
        ''' 得到可比较的样本对，其中生存时间段的在前面 '''
        batch_size = len(status)
        indx = torch.arange(batch_size)
        pairs1 = indx.repeat(batch_size)
        pairs2 = indx.repeat_interleave(batch_size, dim=0)
        # 选择第一个生存时间小于第二个的元素
        time_mask = time[pairs1] < time[pairs2]
        pairs1, pairs2 = pairs1[time_mask], pairs2[time_mask]
        # 选择生存时间小的event必须是1
        event_mask = status[pairs1] == 1
        pairs1, pairs2 = pairs1[event_mask], pairs2[event_mask]
        return pairs1, pairs2

