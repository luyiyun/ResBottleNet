import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def process_ml_result(fn):
    dat = pd.read_csv(fn)
    use_dat = dat.iloc[:, 1:].drop(columns='split_index')
    return use_dat[use_dat['phase'] == 'valid'].drop(columns='phase')


def process_dltest_result(fn, indx=None):
    dat = pd.read_csv(fn)
    use_dat = dat.iloc[:, 1:]
    use_dat = use_dat.drop(columns=[c for c in use_dat.columns if 'Loss' in c])
    use_dat = use_dat.melt(
        'split_index', var_name='score_type', value_name='score_value')
    use_dat['classifier'] = 'dl%s' % indx
    return use_dat


def process_dl_result(fn, indx=None):
    dat = pd.read_csv(fn)
    use_dat = dat.iloc[:, 1:]
    use_dat = use_dat.drop(columns=[c for c in use_dat.columns if 'Loss' in c])
    use_dat = use_dat.melt('split_index')
    use_dat[['score_type', 'phase']] = use_dat['variable'].str.split(
        '_', expand=True)
    use_dat = use_dat.groupby(['score_type', 'phase', 'split_index'])['value']
    if indx is not None:
        use_dat = use_dat.agg(
            {
                'dl%s_peak' % indx: 'max',
                'dl%s_last' % indx: lambda i: i.iloc[-1]
            }
        )
    else:
        use_dat = use_dat.agg(
            {'dl_peak': 'max', 'dl_last': lambda i: i.iloc[-1]})
    use_dat = use_dat.reset_index().drop(columns='split_index')
    use_dat = use_dat.melt(
        ['score_type', 'phase'], var_name='classifier',
        value_name='score_value'
    )
    return use_dat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dirs', nargs='+')
    parser.add_argument('-di', '--dl_index', nargs='+')
    args = parser.parse_args()
    reses = []
    for d, i in zip(args.dirs, args.dl_index):
        for fn in os.listdir(d):
            if fn == 'ml_train.csv':
                res = process_ml_result(os.path.join(d, fn))
                reses.append(res)
            elif fn == 'test.csv':
                res = process_dltest_result(os.path.join(d, fn), i)
                reses.append(res)
    reses = pd.concat(reses)

    # 画图
    sts = reses.score_type.unique()

    fig, axes = plt.subplots(ncols=len(sts), figsize=(25, 5))
    for i, st in enumerate(sts):
        subdf = reses.loc[reses.score_type == st]
        subdf.boxplot(column='score_value', by='classifier', ax=axes[i])
        axes[i].set_title(st)
        for tick in axes[i].get_xticklabels():
            tick.set_rotation(25)
    plt.show()


if __name__ == "__main__":
    main()
