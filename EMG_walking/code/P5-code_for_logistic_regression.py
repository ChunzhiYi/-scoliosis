import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump  # 导入 joblib 以保存模型
from sklearn.preprocessing import StandardScaler

# 1. 加载 .mat 文件
mat_file_path = 'feature_matrix.mat'  # 替换为您的 .mat 文件路径
data = loadmat(mat_file_path)

# 2. 提取 newMatrixall 矩阵
new_matrix_all = data['newMatrixall']

# 3. 分离标签和特征
X = new_matrix_all[:, 1:]  # 特征
y = new_matrix_all[:, 0]    # 标签

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化特征数据

# 4. 创建逻辑回归模型
model = LogisticRegression(max_iter=500)
spliter = StratifiedKFold(n_splits=5, shuffle=True)

# 5. 进行交叉验证并获得预测
y_pred = cross_val_predict(model, X_scaled, y, cv=spliter)  # 5折交叉验证

# 6. 计算混淆矩阵
cm = confusion_matrix(y, y_pred)

# 7. 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 8. 打印混淆矩阵
print("Confusion Matrix:")
print(cm)

# 9. 保存模型
# model_filename = 'logistic_regression_model.joblib'  # 定义保存模型的文件名
# dump(model, model_filename)  # 保存模型
# print(f"Model saved as {model_filename}")

# 10. 拟合模型以获取系数
model.fit(X_scaled, y)

# 11. 计算特征重要性（模型系数的绝对值）
feature_importance = np.abs(model.coef_[0])

# 12. 找出重要性前十特征序号
top_features_indices = np.argsort(feature_importance)[-20:][::-1]  # 排序并取前十个特征

# 13. 打印重要性前十特征的序号和对应的重要性值
print("Top 10 Important Features (Index and Importance):")
for idx in top_features_indices:
    print(f"Feature {idx + 1}: Importance = {feature_importance[idx]}")

# ---- 额外代码：保存标签为0的分类结果为CSV文件 ----

# 找出标签为 0 的样本索引
label_0_indices = np.where(y == 0)[0]

# 获取标签为0的样本的预测值
y_pred_0 = y_pred[label_0_indices]

# 标签为0且预测为1的样本向量
label_0_pred_1 = X_scaled[label_0_indices[y_pred_0 == 1]]

# 标签为0且预测为2的样本向量
label_0_pred_2 = X_scaled[label_0_indices[y_pred_0 == 2]]

# 标签为0且预测正确的样本向量（即预测为0）
label_0_correct = X_scaled[label_0_indices[y_pred_0 == 0]]

# 将这些向量保存为 CSV 文件
# 转换为 DataFrame 以便保存为 CSV
label_0_pred_1_df = pd.DataFrame(label_0_pred_1)
label_0_pred_2_df = pd.DataFrame(label_0_pred_2)
label_0_correct_df = pd.DataFrame(label_0_correct)

# 保存为 CSV 文件
label_0_pred_1_df.to_csv('label_0_pred_1.csv', index=False, header=False)
label_0_pred_2_df.to_csv('label_0_pred_2.csv', index=False, header=False)
label_0_correct_df.to_csv('label_0_correct.csv', index=False, header=False)

print("Saved the vectors as CSV files:")
print("- label_0_pred_1.csv: Vectors where label 0 is predicted as 1")
print("- label_0_pred_2.csv: Vectors where label 0 is predicted as 2")
print("- label_0_correct.csv: Vectors where label 0 is correctly predicted as 0")