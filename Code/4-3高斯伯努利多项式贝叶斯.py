
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))

print("Naive Gausian bayes score (sklearn): " +str(nb.score(X_test, y_test)))


nb = MultinomialNB()
nb.fit(X_train, y_train)

y_pred = nb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))

print("Naive Gausian bayes score (sklearn): " +str(nb.score(X_test, y_test)))


nb = BernoulliNB()
nb.fit(X_train, y_train)

y_pred = nb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"% (iris.data.shape[0],(iris.target != y_pred).sum()))
print("Naive Gausian bayes score (sklearn): " +str(nb.score(X_test, y_test)))

min_x=min(np.min(X_train.ravel()),np.min(X_test.ravel()))-0.1
max_x=max(np.max(X_train.ravel()),np.max(X_test.ravel()))+0.1
binarizes=np.linspace(min_x,max_x,endpoint=True,num=100)

train_scores=[]
test_scores=[]

for binarize in binarizes:
    cls=BernoulliNB(binarize=binarize)
    cls.fit(X_train,y_train)
    train_scores.append(cls.score(X_train,y_train))
    test_scores.append(cls.score(X_test, y_test))

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(binarizes,train_scores,label="Training Score")
ax.plot(binarizes,test_scores,label="Testing Score")
ax.set_xlabel("binarize")
ax.set_ylabel("score")
ax.set_ylim(0,1.0)
ax.set_xlim(min_x-1,max_x+1)
ax.set_title("BernoulliNB")
ax.legend(loc="best")
plt.show()

# 这几个都是naive bayes的模型，区别主要在于特征的分布。
#
#
#
# 如果特征是数值的，最好是正态分布的数值的，那么用
# sklearn.naive_bayes.GaussianNB

# 如果特征是binary的，那么用
# sklearn.naive_bayes.BernoulliNB

# 如果特征是categorical的，那么用
# sklearn.naive_bayes.MultinomialNB