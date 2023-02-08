from sklearn.svm import LinearSVR

regr = LinearSVR(random_state=0, tol=1e-5)
regr.fit(X, y)
print(regr.coef_)

print(regr.intercept_)
print(regr.predict([[1, 1]]))
