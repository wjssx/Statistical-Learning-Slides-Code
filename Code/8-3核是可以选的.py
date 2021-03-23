from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts

#import our data
iris = datasets.load_iris()
X = iris.data
y = iris.target

#split the data to  7:3
X_train,X_test,y_train,y_test = ts(X,y,test_size=0.3)

# select different type of kernel function and compare the score

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf',gamma='auto')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear',gamma='auto')
print("xxxxxxxxxxxxxxxx")
print(X_train.shape)
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly',gamma='auto')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
print("The score of poly is : %f"%score_poly)

print(clf_linear.coef_,clf_linear.intercept_)

print(clf_linear.predict([[4.9,3.,1.4,0.2]]))
# LinearSVC
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts


iris = datasets.load_iris()
X = iris.data
y = iris.target

#split the data to  7:3
X_train,X_test,y_train,y_test = ts(X,y,test_size=0.3)


clf = LinearSVC(random_state=0, tol=1e-5,max_iter = 10000)
clf.fit(X, y)

print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[4.9,3.,1.4,0.2]]))


from sklearn.svm import SVR

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y)
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
clf.predict([[1, 1]])



from sklearn.svm import LinearSVR

regr = LinearSVR(random_state=0, tol=1e-5)
regr.fit(X, y)
print(regr.coef_)

print(regr.intercept_)
print(regr.predict([[1,1]]))

