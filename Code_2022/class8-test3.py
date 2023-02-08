from sklearn import svm
from sklearn.svm import SVR

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y)
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([[1, 1]]))

from sklearn.svm import LinearSVR

regr = LinearSVR(random_state=0, tol=1e-5)
regr.fit(X, y)
print(regr.coef_)

print(regr.intercept_)
print(regr.predict([[1, 1]]))
