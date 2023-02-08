import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.head(5)
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df.label.value_counts()

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
print(data[:10, :])
X, y = data[:, :-1], data[:, -1]
print(X[:10, :])
y = np.array([1 if i == 1 else -1 for i in y])


class PLA_dual:
    def __init__(self, max_iter=1000):
        self.b = 0
        self.lr = 0.1
        self.max_iter = max_iter
        self.iter = 0

    def cal_w(self, X):
        w = 0
        for i in range(len(self.alpha)):
            w += self.alpha[i] * y[i] * X[i]
        return w

    def gram_matrix(self, X):
        return np.dot(X, X.T)

    def fit(self, X, y):
        N, M = X.shape
        self.alpha = np.zeros(N)
        gram = self.gram_matrix(X)
        for n in range(self.max_iter):
            self.iter = n
            wrong_items = 0
            for i in range(N):
                tmp = 0
                for j in range(N):
                    tmp += self.alpha[j] * y[j] * gram[i, j]
                tmp += self.b
                if y[i] * tmp <= 0:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y[i]
                    wrong_items += 1
            if wrong_items == 0:
                self.w = self.cal_w(X)
                print("finished at iters: {}, w: {}, b: {}".format(self.iter, self.w, self.b))
                return
        self.w = self.cal_w(X)
        print("finished for reaching the max_iter: {}, w: {}, b: {}".format(self.max_iter, self.w, self.b))
        return


perceptron3 = PLA_dual()
perceptron3.fit(X, y)


def plot(model, tilte):
    x_points = np.linspace(4, 7, 10)
    y_ = -(model.w[0] * x_points + model.b) / model.w[1]
    plt.plot(x_points, y_)
    print(y_)

    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='-1')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title(tilte)
    plt.legend()
    plt.show()


plot(perceptron3, 'PLA_dual')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

iris = load_iris()
X = iris.data
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# clf=MLPClassifier(activation='logistic',max_iter=1000)# 构造分类器实例
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                    hidden_layer_sizes=(20, 20, 20), random_state=1, max_iter=10000)
# 4*20*20*20个参数

clf.fit(X_train, y_train)  # 训练分类器
print(clf.score(X_test, y_test))  # 查看在训练集上的评价预测精度
