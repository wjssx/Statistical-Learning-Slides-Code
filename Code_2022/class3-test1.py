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


class PLA:
    def __init__(self, max_iter=1000, shuffle=False):
        self.b = 0
        self.lr = 0.1
        self.max_iter = max_iter
        self.iter = 0
        self.shuffle = shuffle

    def sign(self, x, w, b):
        return np.dot(x, w) + b

    def fit(self, X, y):
        N, M = X.shape
        self.w = np.ones(M)
        for n in range(self.max_iter):
            self.iter = n
            wrong_items = 0
            if self.shuffle:  # 每次迭代，是否打乱
                idx = np.random.permutation(range(N))
                X, y = X[idx], y[idx]
            for i in range(N):
                if y[i] * self.sign(X[i], self.w, self.b) <= 0:
                    self.w += self.lr * np.dot(y[i], X[i])
                    self.b += self.lr * y[i]
                    wrong_items += 1
            if wrong_items == 0:
                print("finished at iters: {}, w: {}, b: {}".format(self.iter, self.w, self.b))
                return
        print("finished for reaching the max_iter: {}, w: {}, b: {}".format(self.max_iter, self.w, self.b))
        perceptron1 = PLA()
        perceptron1.fit(X, y)


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


perceptron1 = PLA()
perceptron1.fit(X, y)
plot(perceptron1, 'PLA_dual')
plt.show()

####################################################
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

# import numpy as np
# import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
Y = iris.target

df = pd.DataFrame(iris.data, columns=iris.feature_names)

# df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
df['label'] = iris.target

#
data = np.array(df.iloc[:100, [0, 1, -1]])

x, y = data[:, :-1], data[:, -1]

# print(data)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9)

clf = Perceptron(tol=1e-3, random_state=0, max_iter=1000)

clf.fit(X_train, y_train)

print(clf.coef_)

print(clf.intercept_)

x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

###################################################
from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(clf, X_train, y_train)
disp.figure_.suptitle("Confusion Matrix")
print("Confusion matrix:\n%s" % disp.confusion_matrix)
###################################################

from sklearn.metrics import plot_precision_recall_curve

pr = plot_precision_recall_curve(clf, X_test, y_test)
from sklearn.metrics import plot_roc_curve

roc = plot_roc_curve(clf, X_test, y_test)
