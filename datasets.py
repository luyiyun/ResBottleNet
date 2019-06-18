import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def _from_files_get_prediction(cli_file, rna_file, y_name, **kwargs):
    '''
    cli_file：临床数据文件；
    rna_file：RNAseq数据文件；
    y_name：使用的y的name，可以是多个，也可以是dict，则其key是原始变量名，
        value是更改后的变量名；
    kwargs：用于传递至RnaData的实例化方法的其他参数;
    '''
    # 获取标签
    cli_df = pd.read_csv(cli_file, sep='\t', index_col='sampleID')
    if isinstance(y_name, dict):
        y = cli_df[list(y_name.keys())].dropna().rename(columns=y_name)
    else:
        y = cli_df[[y_name]].dropna()
    y_num = y.shape[-1]
    # 获取RNAseq数据
    rna_df = pd.read_csv(rna_file, sep='\t', index_col=0)
    rna_df = rna_df.T.rename_axis(index='sample', columns='genes')
    # 进行merge
    all_df = y.merge(
        rna_df, how='inner', left_index=True, right_index=True)

    return RnaData(
        all_df.iloc[:, y_num:], all_df.iloc[:, :y_num], **kwargs)


class RnaData:
    def __init__(self, X, Y, label_mapper=None):
        # X和y分开储存，便于之后的操作
        # 将X和Y都变成DataFrame，这样生成的values都是2维的矩阵，便于之后处理
        assert isinstance(X, (pd.DataFrame))
        assert isinstance(Y, (pd.DataFrame, pd.Series))
        self.X_ = X
        self.Y_ = Y if isinstance(Y, pd.DataFrame) else pd.DataFrame(Y)
        if label_mapper is not None:
            self.Y_ = self.Y_.applymap(lambda x: label_mapper[x])
        self.XY_ = pd.concat([self.X_, self.Y_], axis=1)

    def __len__(self):
        return len(self.Y_)

    @property
    def XY(self):
        return self.XY_

    @property
    def Y(self):
        return self.Y_

    @property
    def X(self):
        return self.X_

    @property
    def all_gene_names(self):
        return list(self.X_.columns)

    def transform(self, func):
        self.X_, self.Y_ = func(self.X_, self.Y_)
        return self.X_, self.Y_

    @staticmethod
    def predicted_data(cli_file, rna_file, y_name, **kwargs):
        '''
        cli_file：临床数据文件；
        rna_file：RNAseq数据文件；
        y_name：使用的y的name，可以是多个，也可以是dict，则其key是原始变量名，
            value是更改后的变量名；
        kwargs：用于传递至RnaData的实例化方法的其他参数;
        '''
        return _from_files_get_prediction(cli_file, rna_file, y_name, **kwargs)

    @staticmethod
    def survival_data(cli_file, rna_file, status_name, time_name, **kwargs):
        res = _from_files_get_prediction(
            cli_file, rna_file, {status_name: 'status', time_name: 'time'},
            **kwargs
        )
        res.Y_ = res.Y_.reindex(columns=['status', 'time'])
        return res

    def split_cv(
        self, test_size, n_split, stratify=None, shuffle=True, seed=1234
    ):
        '''
        如果n_split不是None，则使用的是stratified shuffle split，得到一个返回
        RnaData pair的iterator
        '''
        if stratify is None:
            stra_y = self.Y_.iloc[:, 0].values
        else:
            stra_y = self.Y_[stratify].values
        sss = StratifiedShuffleSplit(
            n_split, test_size=test_size, random_state=seed)
        for train_index, test_index in sss.split(self.X_.values, stra_y):
            X_train = self.X_.iloc[train_index, :]
            X_test = self.X_.iloc[test_index, :]
            Y_train = self.Y_.iloc[train_index, :]
            Y_test = self.Y_.iloc[test_index, :]
            yield RnaData(X_train, Y_train), RnaData(X_test, Y_test)

    def split(self, test_size, stratify=None, shuffle=True, seed=1234):
        '''
        将此RnaData分成2个RnaData，用于train、test数据集分割
        '''
        if stratify is None:
            stra_y = self.Y_.iloc[:, 0].values
        else:
            stra_y = self.Y_[stratify].values
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X_.values, self.Y_.values, test_size=test_size,
            random_state=seed, shuffle=shuffle, stratify=stra_y
        )
        X_train = pd.DataFrame(X_train, columns=self.X_.columns)
        X_test = pd.DataFrame(X_test, columns=self.X_.columns)
        Y_train = pd.DataFrame(Y_train, columns=self.Y_.columns)
        Y_test = pd.DataFrame(Y_test, columns=self.Y_.columns)
        return RnaData(X_train, Y_train), RnaData(X_test, Y_test)

    def save(self, fn):
        torch.save(self, fn)

    @staticmethod
    def load(fn):
        return torch.load(fn)

    def to_torchdat(self):
        return torch.utils.data.TensorDataset(
            torch.tensor(self.X_.values, dtype=torch.float),
            torch.tensor(self.Y_.values.squeeze())
        )


def test():
    import sys
    import time

    brca_root = "E:/Python/code_genomics/pytorch/Attention/DATA/BRCA"
    brca_cli = os.path.join(brca_root, 'BRCA_clinicalMatrix')
    brca_rna = os.path.join(brca_root, 'HiSeqV2')
    pan_root = "E:/Python/code_genomics/pytorch/Attention/DATA/Pan"
    pan_cli = os.path.join(pan_root, 'PANCAN_clinicalMatrix')
    pan_rna = os.path.join(pan_root, 'HiSeqV2')
    if sys.argv[1] == 'brca':
        t1 = time.perf_counter()
        rna_data = RnaData.predicted_data(
            brca_cli, brca_rna, 'PAM50Call_RNAseq')
        print(rna_data.X.head())
        print(rna_data.Y.head())
        print(rna_data.all_gene_names[:5])
        print(len(rna_data.all_gene_names))
        # transform 方法测试
        # x, y = rna_data.transform(lambda x, y: (x * 2, y))
        # print(x.head())
        # print(y.head())
        # print(rna_data.X.head())
        # print(rna_data.Y.head())
        x, y = rna_data.transform(lambda x, y: (x.iloc[:, :5], y))
        print(x.head())
        print(y.head())
        print(rna_data.X.head())
        print(rna_data.Y.head())

        # split方法测试
        stra_y = "PAM50Call_RNAseq"
        train, test = rna_data.split(0.2, stra_y)
        print(len(train))
        print(len(test))
        print(train.X.head())
        print(test.X.head())
        print(train.Y.head())
        print(test.Y.head())
        print(test.Y.iloc[:, 0].value_counts())

        # for i, (train, test) in enumerate(
        #     rna_data.split_cv(0.2, stra_y, n_split=10)
        # ):
        #     print(i)
        #     print(len(train))
        #     print(len(test))
        t2 = time.perf_counter()
        print(t2 - t1)
    elif sys.argv[1] == 'survival':
        t1 = time.perf_counter()
        rna_data = RnaData.survival_data(pan_cli, pan_rna, '_OS_IND', '_OS')
        print(rna_data.X.head())
        print(rna_data.Y.head())
        print(rna_data.all_gene_names[:5])
        print(len(rna_data.all_gene_names))
        print(rna_data.X.shape)
        print(rna_data.Y.shape)

        rna_data.save(
            'E:/Python/code_genomics/pytorch/Attention/DATA/temp_pan.pth')

        t2 = time.perf_counter()
        print(t2 - t1)


if __name__ == "__main__":
    test()
