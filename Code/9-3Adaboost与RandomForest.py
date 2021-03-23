from sklearn import datasets

digits = datasets.load_digits()
from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)





from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, max_depth=2,random_state=0) #用了三百棵树
forest.fit(X_train, y_train)

in_score = forest.score(X_train, y_train)
out_score = forest.score(X_test, y_test)
print(in_score,out_score)

from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
clf.fit(X_train, y_train)
in_score = clf.score(X_train, y_train)
out_score = clf.score(X_test, y_test)
print(in_score,out_score)

