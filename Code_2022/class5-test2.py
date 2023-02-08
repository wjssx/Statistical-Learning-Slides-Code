from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB, MultinomialNB # 伯努利模型和多项式模型

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

print(nb)