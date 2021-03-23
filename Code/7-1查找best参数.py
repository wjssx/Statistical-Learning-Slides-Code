from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV #找离散变量中最好的参数
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

parameters = {'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), 'C':[0.1,1,2]}
print(parameters)
classifier = LogisticRegression(max_iter=3000)


clf = GridSearchCV(classifier, parameters)
clf.fit(iris.data, iris.target)

sorted(clf.cv_results_.keys())

print(clf.best_params_)
print(clf.best_score_)

from sklearn.model_selection import RandomizedSearchCV  #找连续变量中

clf = RandomizedSearchCV(classifier, parameters)

clf.fit(iris.data, iris.target)

sorted(clf.cv_results_.keys())

print(clf.best_params_)
print(clf.best_score_)