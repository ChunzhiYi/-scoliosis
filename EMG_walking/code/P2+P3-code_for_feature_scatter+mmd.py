import scipy.io as scio
import numpy as np
import pandas as pd

"""
需要做的事情，每种运动单独做，
1.整合特征，把特征拼接成一个矩阵X，行数是实验对象数量，列数是特征数；
2.打标签，把群中表格上的颜色转成标签，标签是一个列向量Y，表示对应实验对象属于哪一类
3.应用下面的代码，得出分析结果
"""


"""降维与可视化"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#
data = scio.loadmat('EMG_feature1104.mat')
features = data['Matrixall_nolabel']

# 创建PCA（主成分分析）模型
model = PCA(n_components=2)
embeddings = model.fit_transform(features)
#
# #labels = np.random.randint(0, 3, len(features))  # 示例随机生成标签
#
# df = pd.DataFrame(embeddings)
# # 指定输出文件路径
# output_file = 'embeddings11.csv'
#
# # 将DataFrame保存为CSV文件
# df.to_csv(output_file, index=False)

data = scio.loadmat('pca_result1104.mat')
embeddings = data['PCA_result']
labels = embeddings[:,2]
embeddings = embeddings[:,:2]

for c in range(3):
    plt.scatter(embeddings[labels == c, 0], embeddings[labels == c, 1], label=c)

plt.legend()

plt.show()

"""衡量特征相似性"""
"""衡量特征相似性"""
from np_mmd import mmd
import random

# 设置随机种子确保结果可复现
np.random.seed(42)
random.seed(42)

# 每组样本数量
sample_size = 1000
# 采样次数
bootstrap_iterations = 100

# 初始化存储每次采样的距离结果
distance1_list = []
distance2_list = []
distance3_list = []

# 获取各组样本
group0 = features[labels == 0]
group1 = features[labels == 1]
group2 = features[labels == 2]

# 进行bootstrap采样和MMD计算
for i in range(bootstrap_iterations):
    # 有放回采样
    idx0 = np.random.choice(len(group0), sample_size, replace=True)
    idx1 = np.random.choice(len(group1), sample_size, replace=True)
    idx2 = np.random.choice(len(group2), sample_size, replace=True)

    # 提取采样后的样本
    sample0 = group0[idx0]
    sample1 = group1[idx1]
    sample2 = group2[idx2]

    # 计算MMD距离
    d1 = mmd(sample0, sample1)
    d2 = mmd(sample0, sample2)
    d3 = mmd(sample1, sample2)

    # 存储结果
    distance1_list.append(d1)
    distance2_list.append(d2)
    distance3_list.append(d3)

    # 打印进度
    if (i + 1) % 100 == 0:
        print(f"完成 {i + 1}/{bootstrap_iterations} 次采样")

# 计算平均距离和标准差
avg_distance1 = np.mean(distance1_list)
avg_distance2 = np.mean(distance2_list)
avg_distance3 = np.mean(distance3_list)

std_distance1 = np.std(distance1_list)
std_distance2 = np.std(distance2_list)
std_distance3 = np.std(distance3_list)

print('Bootstrap采样(1000次)后用MMD距离衡量特征相似度:')
print(f'类别0与类别1: 平均距离 = {avg_distance1:.6f}, 标准差 = {std_distance1:.6f}')
print(f'类别0与类别2: 平均距离 = {avg_distance2:.6f}, 标准差 = {std_distance2:.6f}')
print(f'类别1与类别2: 平均距离 = {avg_distance3:.6f}, 标准差 = {std_distance3:.6f}')

# 导出所有bootstrap采样的距离数据
try:
    # 创建一个DataFrame来保存所有距离数据
    distance_df = pd.DataFrame({
        '迭代次数': range(1, bootstrap_iterations + 1),
        '类别0与类别1_MMD': distance1_list,
        '类别0与类别2_MMD': distance2_list,
        '类别1与类别2_MMD': distance3_list
    })

    # 保存为CSV文件
    distance_df.to_csv('mmd_distances_bootstrap.csv', index=False, encoding='utf-8')
    print("Bootstrap采样的距离数据已成功保存到 'mmd_distances_bootstrap.csv' 文件中。")

    # 同时保存为MAT文件
    scio.savemat('mmd_distances_bootstrap.mat', {
        'distance1_bootstrap': np.array(distance1_list),
        'distance2_bootstrap': np.array(distance2_list),
        'distance3_bootstrap': np.array(distance3_list)
    })
    print("Bootstrap采样的距离数据已成功保存到 'mmd_distances_bootstrap.mat' 文件中。")

    # 保存平均值和标准差
    stats_df = pd.DataFrame({
        '距离类型': ['类别0与类别1', '类别0与类别2', '类别1与类别2'],
        '平均距离': [avg_distance1, avg_distance2, avg_distance3],
        '标准差': [std_distance1, std_distance2, std_distance3]
    })

    stats_df.to_csv('mmd_distances_stats.csv', index=False, encoding='utf-8')
    print("统计数据已成功保存到 'mmd_distances_stats.csv' 文件中。")
except Exception as e:
    print(f"导出数据时出错: {e}")
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# print('用logistic regression模型做分类和特征筛选的分析')
# model = LogisticRegression(max_iter=500)
#
# print('用五折交叉验证判断模型在数据上的分类效果')
# scores = cross_val_score(model, features, labels, cv=5)
# print(f'五折交叉验证准确率：{scores}')
#
#
# print('根据logistic regression的系数判断特征重要性')
# model.fit(features, labels)
# print('拟合准确率：', model.score(features, labels))
# # 系数的绝对值代表重要程度
# print('模型系数：', np.abs(model.coef_))
#
#
# print('计算特征重要性，用Permutation importance')
# from sklearn.inspection import permutation_importance
#
# # 将数据集分为训练集和测试集
# idx = np.arange(len(labels))
# np.random.shuffle(idx)
#
# X_train = features[idx[:round(len(idx)*0.8)]]
# X_test = features[idx[round(len(idx)*0.8):]]
#
# Y_train = labels[idx[:round(len(idx) * 0.8)]]
# Y_test = labels[idx[round(len(idx) * 0.8):]]
#
# # 拟合模型
# model.fit(X_train, Y_train)
# print('用于计算permutation importance的模型在测试集上的效果：')
# print(f'测试准确率：{model.score(X_test, Y_test)}')
#
# perm_imp = permutation_importance(model, X_test, Y_test)
# print('特征重要性结果：')
# print(perm_imp)

"""分类，特征筛选"""
#
