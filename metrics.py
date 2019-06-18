import numpy as np
import torch
from torchnet.meter.meter import Meter
from scipy.special import softmax

from sklearn import metrics
from sksurv.metrics import concordance_index_censored


class Loss(Meter):
    def __init__(self):
        super(Loss, self).__init__()
        self.reset()

    def reset(self):
        self.running_loss = 0.
        self.num_samples = 0

    def add(self, batch_loss, batch_size):
        self.running_loss += batch_loss * batch_size
        self.num_samples += batch_size

    def value(self):
        return self.running_loss / self.num_samples


class SklearnMeter(Meter):
    def __init__(self, func, tensor=True, proba2int=True):
        super(SklearnMeter, self).__init__()
        self.proba2int = proba2int
        self.func = func
        self.tensor = tensor
        self.reset()

    def __call__(self, output, target):
        if self.tensor:
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        if self.proba2int:
            output = output.argmax(-1)
        return self.func(target, output)

    def reset(self):
        self.outputs = []
        self.targets = []

    def add(self, output, target):
        if self.tensor:
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
        self.outputs.append(output)
        self.targets.append(target)

    def value(self):
        self.outputs = np.concatenate(self.outputs, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        if self.proba2int:
            self.outputs = self.outputs.argmax(-1)
        return self.func(self.targets, self.outputs)


class Accuracy(SklearnMeter):
    def __init__(self, proba2int=True, tensor=True, **kwargs):
        def func(y_true, y_pred):
            return metrics.accuracy_score(y_true, y_pred, **kwargs)
        super(Accuracy, self).__init__(func, tensor, proba2int)


class BalancedAccuracy(SklearnMeter):
    def __init__(self, proba2int=True, tensor=True, **kwargs):
        def func(y_true, y_pred):
            return metrics.balanced_accuracy_score(y_true, y_pred, **kwargs)
        super(BalancedAccuracy, self).__init__(func, tensor, proba2int)


class F1Score(SklearnMeter):
    def __init__(self, proba2int=True, tensor=True, **kwargs):
        def func(y_true, y_pred):
            return metrics.f1_score(y_true, y_pred, **kwargs)
        super(F1Score, self).__init__(func, tensor, proba2int)


class Precision(SklearnMeter):
    def __init__(self, proba2int=True, tensor=True, **kwargs):
        def func(y_true, y_pred):
            return metrics.precision_score(y_true, y_pred, **kwargs)
        super(Precision, self).__init__(func, tensor, proba2int)


class Recall(SklearnMeter):
    def __init__(self, proba2int=True, tensor=True, **kwargs):
        def func(y_true, y_pred):
            return metrics.recall_score(y_true, y_pred, **kwargs)
        super(Recall, self).__init__(func, tensor, proba2int)


class ROCAUC(SklearnMeter):
    def __init__(self, tensor=True, score2proba=False, **kwargs):
        def func(y_true, y_pred):
            if 'average' in kwargs:
                # 将y_true变成one-hot向量
                max_num = y_true.max()
                eye_matrix = np.eye(max_num+1)
                y_true = eye_matrix[y_true]
            if score2proba:
                y_pred = softmax(y_pred, axis=1)
            return metrics.roc_auc_score(y_true, y_pred, **kwargs)
        super(ROCAUC, self).__init__(func, tensor, proba2int=False)


class CIndex(SklearnMeter):
    def __init__(self, tensor=True, hazard=True, **kwargs):
        '''
        hazard如果是True，表示这里接受的y_pred是预测的风险，如果是False表示
        预测的是生存
        '''
        def func(y_true, y_pred):
            status, time = y_true[:, 0], y_true[:, 1]
            status = status.astype('bool')
            if not hazard:
                y_pred = -y_pred
            return concordance_index_censored(status, time, y_pred)[0]
        super(CIndex, self).__init__(func, tensor, proba2int=False)
