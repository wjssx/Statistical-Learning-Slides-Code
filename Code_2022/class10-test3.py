from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 加载数据集，是一个字典类似Java中的map
lris_df = datasets.load_iris()

# 挑选出前两个维度作为x轴和y轴，你也可以选择其他维度
x_axis = lris_df.data[:, 0]
y_axis = lris_df.data[:, 2]

# 这里已经知道了分3类，其他分类这里的参数需要调试
model = KMeans(n_clusters=3)

# 训练模型
model.fit(lris_df.data)

# 选取行标为100的那条数据，进行预测
prddicted_label = model.predict([[6.3, 3.3, 6, 2.5]])

# 预测全部150条数据
all_predictions = model.predict(lris_df.data)

# 打印出来对150条数据的聚类散点图
plt.scatter(x_axis, y_axis, c=all_predictions)
plt.show()

