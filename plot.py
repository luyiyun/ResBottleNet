import os
import sys
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def process_ml_result(fn):
    dat = pd.read_csv(fn)
    use_dat = dat.iloc[:, 1:]
    use_dat = use_dat.melt(
        ['split_index', 'phase', 'classifier'],
        var_name='score_type', value_name='score_value'
    )
    return use_dat[use_dat['phase'] == 'test'].drop(
        columns=['phase', 'split_index'])


def process_dltest_result(fn, indx=None):
    dat = pd.read_csv(fn)
    use_dat = dat.iloc[:, 1:]
    use_dat = use_dat.drop(columns=['Loss'])
    use_dat = use_dat.melt(
        'split_index', var_name='score_type', value_name='score_value')
    use_dat['classifier'] = 'dl%s' % indx
    return use_dat.drop(columns='split_index')


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


def get_config(d, *config_names):
    with open(os.path.join(d, 'config.json'), 'r') as f:
        f_content = json.load(f)
        res = [str(f_content[cn]) for cn in config_names]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ds', '--dirs', nargs='+', default=None,
        help="需要图像化的文件夹名称，可以多个"
    )
    parser.add_argument(
        '-di', '--dl_index', default='block_num',
        help=(
            "对于深度学习的方法起使用的后缀，来自config，默认是block_num"
        )
    )
    parser.add_argument(
        '-dsf', '--dirs_file', default=None,
        help="是ds参数组成的txt文件，如果要输入的ds比较多的时候使用"
    )
    parser.add_argument(
        '-fs', '--fig_size', default=(5, 5), nargs=2, type=int,
        help="画图的大小，默认是5x5"
    )
    parser.add_argument(
        '-dc', '--dl_config', default=None, nargs="+",
        help=(
            "要加入的比较的config，默认None，可以写多个，这个在"
            "图像中使用不同颜色来表现"
        )
    )
    args = parser.parse_args()
    if args.dirs_file is not None:
        # 读取txt文件，没一行是一次训练保存结果的文件夹，最后可能有换行符
        with open(args.dirs_file, 'r') as f:
            use_dirs = f.readlines()
            use_dirs = [s.strip('\n') for s in use_dirs]
    else:
        use_dirs = args.dirs
    reses = []
    for d in use_dirs:
        for fn in os.listdir(d):
            if fn == 'ml_train.csv':
                res = process_ml_result(os.path.join(d, fn))
                res['config'] = "ML_no_config"
            elif fn == 'test.csv':
                # 把所有要用的配置都取出
                configs = [args.dl_index]
                if args.dl_config is not None:
                    configs += list(args.dl_config)
                configs = get_config(d, *configs)
                # 后面的配置用于展现在color上
                config_value = '-'.join(configs[1:])
                # 第一个配置的数字用于跟在dl后作为x轴
                res = process_dltest_result(os.path.join(d, fn), configs[0])
                res['config'] = config_value
            reses.append(res)
    reses = pd.concat(reses)

    # 画图
    sts = reses.score_type.unique()

    # fig, axes = plt.subplots(nrows=len(sts), figsize=args.fig_size)
    for i, st in enumerate(sts):
        ax = plt.subplot(
        # if isinstance(axes, np.ndarray):
        #     ax = axes[i]
        # else:
        #     ax = axes
        subdf = reses.loc[reses.score_type == st]
        sns.boxplot(
            'classifier', 'score_value', hue="config", data=subdf, ax=ax)
        ax.set_title(st)
        for tick in ax.get_xticklabels():
            tick.set_rotation(25)
        plt.show()


if __name__ == "__main__":
    main()
