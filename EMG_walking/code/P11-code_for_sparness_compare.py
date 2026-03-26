import scipy.io as scio
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def sparseness(W):
    # 协同数应处于第一个维度
    # W: (协同数,肌肉数)
    spars = []
    for w in W:
        spar = (np.sqrt(len(w)) - np.sum(np.abs(w))/np.sqrt(np.sum(w ** 2)))/(np.sqrt(len(w)) - 1)
        spars.append(spar)
    return np.average(spars)


data = scio.loadmat(r'EMG_sparseness_xzh.mat')

for motion in ["xingzou"]:
    sparseness_w = data[motion][:, -1]
    sparseness_c = data[motion][:, -2]
    labels = data[motion][:, 1].astype(int)
    labels_str = [['small', 'health', 'large'][i] for i in labels]
    data_w = pd.DataFrame({
        'values': sparseness_w,
        'group': labels_str,
    })
    data_c = pd.DataFrame({
        'values': sparseness_c,
        'group': labels_str,
    })

    # 进行多重比较检验（Tukey's HSD）
    tukey_w = pairwise_tukeyhsd(endog=data_w['values'], groups=data_w['group'], alpha=0.05)
    print(f'{motion} Sparseness of W')
    print(tukey_w)
    # 进行多重比较检验（Tukey's HSD）
    tukey_c = pairwise_tukeyhsd(endog=data_c['values'], groups=data_c['group'], alpha=0.05)
    print(f'{motion} Sparseness of C')
    print(tukey_c)

    plt.figure(figsize=(10, 6))
    sns.boxplot(hue='group', y='values', data=data_w, palette='Set3')
    plt.title('sparseness of W')
    plt.savefig(f'{motion}-sparseness-w.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(hue='group', y='values', data=data_c, palette='Set3')
    plt.title('sparseness of C')
    plt.savefig(f'{motion}-sparseness-c.png')
    plt.close()

