from random import shuffle
from tqdm import tqdm  # 导入 tqdm 库以实现进度条

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump , load # 导入 joblib 以保存模型
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. 加载 .mat 文件
mat_file_path = 'feature_matrix_per.mat'  # 替换为您的 .mat 文件路径
data = loadmat(mat_file_path)

# 2. 提取 newMatrixall 矩阵
new_matrix_all = data['newMatrixall']

# 3. 分离标签和特征
X = new_matrix_all[:, 1:]  # 特征（除第一列外）
y = new_matrix_all[:, 0]  # 标签（第一列）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化特征数据

model = LogisticRegression(max_iter=500)
spliter = StratifiedKFold(n_splits=5, shuffle=True)
# original_accuracy = cross_val_score(model,  X_scaled, y, cv=spliter)
# original_accuracy = np.mean(original_accuracy)
original_accuracy = 0.45
# 6. 进行Permutation Test
n_permutations = 100
p_values = []
importance_drops = []

# 使用 tqdm 包装循环以显示进度条
for feature_index in range(X_scaled.shape[1]):
    accuracies = []

    # 打乱特征
    for n in tqdm(range(n_permutations)):
        X_permuted = X_scaled.copy()
        np.random.shuffle(X_permuted[:, feature_index])  # 打乱当前特征列
        permuted_accuracy = cross_val_score(model, X_permuted, y, cv=5).mean()
        accuracies.append(permuted_accuracy)

    # 计算 p 值
    p_value = np.sum(np.array(accuracies) < original_accuracy) / n_permutations
    p_values.append(p_value)

    # 计算准确率下降
    importance_drop = original_accuracy - np.mean(accuracies)
    importance_drops.append(importance_drop)

# 7. 找出显著特征（p < 0.05）
significant_features = np.array(p_values) < 0.05

# 8. 计算显著特征的排序
feature_importance = np.array(importance_drops)[significant_features]
top_feature_indices = np.argsort(feature_importance)[-10:][::-1]  # 排序并取前十个特征

# 9. 打印重要性前十特征的序号和对应的下降幅度
print("Top 10 Important Features (Index and Importance Drop):")
if len(top_feature_indices) > 0:
    for idx in top_feature_indices:
        print(f"Feature {idx + 1}: Importance Drop = {importance_drops[idx]}")
else:
    print("没有显著特征。")

# 10. 计算并显示混淆矩阵
y_pred = cross_val_predict(model, X_scaled, y, cv=5)  # 5折交叉验证
cm = confusion_matrix(y, y_pred)

# 11. 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
