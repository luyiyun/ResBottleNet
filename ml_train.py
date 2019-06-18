import os
import time
import copy

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM, FastSurvivalSVM
from sksurv.util import Surv

from config import config
from datasets import RnaData
import metrics as mm
import transfers as tf


def default_classical_scorings(task="predict"):
    if task == 'predict':
        scorings = (
            mm.Accuracy(tensor=False), mm.BalancedAccuracy(tensor=False),
            mm.F1Score(average='macro', tensor=False),
            mm.Precision(average='macro', tensor=False),
            mm.Recall(average='macro', tensor=False),
            mm.ROCAUC(average='macro', tensor=False)
        )
    else:
        scorings = (mm.CIndex(tensor=False, hazard=True),)
    return scorings


class Timing:
    '''
    一个上下文管理器，用于记录在此管理器下运行程序所花费的时间(self.duration)
    '''
    unit_dict = {
        's': 1,
        'm': 60,
        'h': 300
    }

    def __init__(self, unit='s'):
        self.duration = None
        self.unit_num = self.unit_dict[unit]

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        self.duration = (self.end - self.start) / self.unit_num


class ClassicialML:
    '''
    这是一个基类，可以对多个模型进行训练测试并保存结果
    每一种模型使用一个类来表示，每个类均存在一个训练用方法和预测用方法，
    此类的主要功能是对所有的模型进行训练及预测，并将结果记录，其主要过程是通过
    方法fit_eval来完成的，其主要过程是:
        1. 使用输入的trainx和trainy进行训练；
        2. 之后在trainx,testx上进行预测；
        3. 使用预测得到的结果和trainy、testy进行评价，计算score；
        4. 结果保存在self.results中；
    对于不同类型的研究（predict或survival），可能其模型的训练方法、预测方法、以
        及使用的数据形式不同，所以继承此类时需要主要根据不同的研究实现这些东西
    '''
    def __init__(
        self, estimators, scorings, fit_name='fit',
        predict_name='predict_proba', verbose=True
    ):
        '''
        estimators: 使用的模型的实例；
        scorings: 评价方法的实例，必须是callable对象；
        fit_name: 模型其训练的方法名；
        predict_name: 模型其预测的方法名；
        verbose: True则会在训练的时候打印信息；
        '''
        self.estimators = estimators
        self.clear_estimators = copy.deepcopy(estimators)
        self.scorings = scorings
        self.fit_name = fit_name
        self.predict_name = predict_name
        self.verbose = verbose

    def fit_eval(self, trainX, trainy, testX, testy):
        ''' 这里接受的4个参数都是df '''
        fit_params = self._fit_params(trainX, trainy)
        predict_params_es = [
            (p, self._predict_params(X), y)
            for X, p, y in zip(
                [trainX, testX], ['train', 'test'], [trainy, testy]
            )
        ]
        self.results = []
        for estimator in self.estimators:
            fit_method = getattr(estimator, self.fit_name)
            predict_method = getattr(estimator, self.predict_name)
            if self.verbose:
                print(self._verbose(estimator, phase='ALL') + ', beginning!!!')
            with Timing() as fit_timing:
                fit_method(**fit_params)
            print(
                self._verbose(
                    estimator, phase='train',
                    scores={'Time': fit_timing.duration}
                )
            )
            for p, ps, y in predict_params_es:
                pred = predict_method(**ps)
                one_evaluation = {}
                for m in self.scorings:
                    score = m(**self._score_params(pred, y))
                    one_evaluation[m.__class__.__name__] = score
                if self.verbose:
                    print(
                        self._verbose(
                            estimator, phase=p, scores=one_evaluation)
                    )
                one_evaluation['phase'] = p
                one_evaluation['classifier'] = estimator.__class__.__name__
                # one_evaluation['traing_time'] = fit_timing.duration
                self.results.append(copy.deepcopy(one_evaluation))
        self.results = pd.DataFrame(self.results)

    @staticmethod
    def _verbose(estimator, phase="train", scores=None):
        ''' 将正在训练的信息打印 '''
        string = "phase: %s, classifier: %s" % (
            phase, estimator.__class__.__name__)
        if scores is not None:
            strs = []
            for k, v in scores.items():
                strs.append("%s: %.4f" % (k, v))
            scoring_string = ', '.join(strs)
            string = string + ', ' + scoring_string
        return string

    def reset(self):
        self.estimators = copy.deepcopy(self.clear_estimators)

    def _fit_params(self, X, y):
        '''
        输入的是df格式的X和y，输出的是dict，作为fit方法的kwargs
        '''
        raise NotImplementedError

    def _predict_params(self, X):
        '''
        输入的是df格式的X，输出的是dict，用于predict方法的kwargs
        '''
        raise NotImplementedError

    def _score_params(self, predict, y):
        '''
        输入的是模型的输出predict和df格式的y，输出是用于各种scoring方法的
        输入
        '''
        raise NotImplementedError


class PredictedML(ClassicialML):
    def __init__(self, scorings=default_classical_scorings('predict')):
        estimators = [
                RandomForestClassifier(n_estimators=100),
                SVC(gamma='scale', probability=True),  # 设定gamma为了避免warning
                MultinomialNB(),
                KNeighborsClassifier()
        ]
        super(PredictedML, self).__init__(
            estimators, scorings, 'fit', 'predict_proba', verbose=True)

    def _fit_params(self, X, y):
        return {'X': X.values, 'y': y.values.squeeze()}

    def _predict_params(self, X):
        return {'X': X.values}

    def _score_params(self, pred, y):
        return {
            'output': pred.squeeze(), 'target': y.values.squeeze()}


class SurvivalML(ClassicialML):
    def __init__(self, scorings=default_classical_scorings('survival')):
        estimators = [
            # CoxPHSurvivalAnalysis(verbose=1),
            ComponentwiseGradientBoostingSurvivalAnalysis(verbose=1),
            GradientBoostingSurvivalAnalysis(verbose=1),
            FastKernelSurvivalSVM(kernel='rbf', verbose=1),
            FastSurvivalSVM(verbose=1)
        ]
        super(SurvivalML, self).__init__(
            estimators, scorings, 'fit', 'predict', verbose=True)

    def _fit_params(self, X, y):
        y = Surv.from_dataframe('status', 'time', y)
        return {'X': X.values, 'y': y}

    def _predict_params(self, X):
        return {'X': X.values}

    def _score_params(self, pred, y):
        return {'output': pred.squeeze(), 'target': y.values}


def main():

    # ----- 根据data来读取不同的数据和使用不同的对象、loss、metrics -----
    if config.args.data == 'brca':
        rna = RnaData.predicted_data(
            config.brca_cli, config.brca_rna,
            {'PAM50Call_RNAseq': 'pam50'}
        )
        rna.transform(tf.LabelMapper(config.brca_label_mapper))
        ml = PredictedML()
    elif config.args.data == 'survival':
        if os.path.exists('./DATA/temp_pan.pth'):
            rna = RnaData.load('./DATA/temp_pan.pth')
        else:
            rna = RnaData.survival_data(
                config.pan_cli, config.pan_rna, '_OS_IND', '_OS')
        ml = SurvivalML()
    rna.transform(tf.ZeroFilterCol(0.8))
    rna.transform(tf.MeanFilterCol(1))
    rna.transform(tf.StdFilterCol(0.5))

    # ----- 训练模型 -----
    split_iterator = rna.split_cv(
        config.args.test_size, config.args.cross_valid)
    hists = []
    for split_index, (train_rna, test_rna) in enumerate(split_iterator):
        print(
            '##### save: %s, split: %d #####' %
            (config.args.save, split_index)
        )
        ml.fit_eval(
            train_rna.X, train_rna.Y,
            test_rna.X, test_rna.Y
        )
        res = ml.results
        res['split_index'] = split_index
        hists.append(res)

    hists = pd.concat(hists)
    hists.to_csv(os.path.join(config.save_dir, 'ml_train.csv'))


if __name__ == "__main__":
    main()
