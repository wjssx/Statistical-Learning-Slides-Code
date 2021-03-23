import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
#%matplotlib inline
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
print(df.head(10))
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df.label.value_counts())
data = np.array(df.iloc[:100, [0, 1, -1]])
print(data[:10,:])
print()

X, y = data[:,:-1], data[:,-1]
X[:10,:]
y = np.array([1 if i == 1 else -1 for i in y ])
#clf = Perceptron(fit_intercept=False, shuffle=False)
clf = Perceptron(tol=1e-3, random_state=0, max_iter=1000)
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

