import os

import pandas as pd
from datasets import RnaData
import sklearn.preprocessing as skp


class LabelMapper:
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, x, y):
        y = y.applymap(lambda x: self.mapper[x]).astype('int64')
        return x, y


class ZeroFilterCol:
    def __init__(self, zero_frac=0.8):
        self.zero_frac = zero_frac

    def __call__(self, x, y):
        mask = (x == 0).mean(axis=0) < self.zero_frac
        return x.loc[:, mask], y


class MeanFilterCol:
    def __init__(self, mean_thre=1):
        self.mean_thre = mean_thre

    def __call__(self, x, y):
        mask = x.mean(axis=0) > self.mean_thre
        return x.loc[:, mask], y


class StdFilterCol:
    def __init__(self, std_thre=0.5):
        self.std_thre = std_thre

    def __call__(self, x, y):
        mask = x.std(axis=0) > self.std_thre
        return x.loc[:, mask], y


class Normalization:
    Scalers = {
        'standard': skp.StandardScaler,
        'minmax': skp.MinMaxScaler,
        'maxabs': skp.MaxAbsScaler,
        'robust': skp.RobustScaler
    }

    def __init__(self, ty='standard', **kwargs):
        self.ty = ty
        self.scaler = __class__.Scalers[ty](**kwargs)
        self.fit_ind = False

    def __call__(self, x, y):
        xindex, xcolumns = x.index, x.columns  # scaler的结果是ndarray，但需要df
        if self.fit_ind:
            x = self.scaler.transform(x.values)
            return pd.DataFrame(x, index=xindex, columns=xcolumns), y
        else:
            x = self.scaler.fit_transform(x.values)
            self.fit_ind = True
            return pd.DataFrame(x, index=xindex, columns=xcolumns), y

    def reset(self):
        if self.fit_ind:
            self.scaler = __class__.Scalers[self.ty]
            self.fit_ind = False


def test():
    import sys

    gene_root = "G:/Analysis/DLforGenomic/genomic_data"
    cli_file = os.path.join(gene_root, 'BRCA/BRCA_clinicalMatrix')
    rna_file = os.path.join(gene_root, 'BRCA/HiSeqV2')

    rna = RnaData.from_files(cli_file, rna_file, {'PAM50Call_RNAseq': 'pam50'})
    if sys.argv[1] == 'FilterCol':
        print(rna.X.shape)
        rna.transform(ZeroFilterCol(0.8))
        print(rna.X.shape)
        rna.transform(MeanFilterCol(1))
        print(rna.X.shape)
        rna.transform(StdFilterCol(0.5))
        print(rna.X.shape)


if __name__ == "__main__":
    test()
