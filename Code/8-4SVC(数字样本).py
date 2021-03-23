from sklearn import datasets, svm, metrics


digits = datasets.load_digits()
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


clf = svm.SVC()

clf.fit(X_train, y_train,)

print(clf.score(X_test, y_test))

