import pandas as  pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
fish= pd.read_csv("fish-01.csv")
X_train, X_test, y_train, y_test = train_test_split(fish.iloc[:,1:], fish.iloc[:,0], test_size=0.4, random_state=0)

clf = GaussianNB(priors=None, var_smoothing=1e-09)
clf.fit(X_train, y_train)
print(clf.predict([[120.0, 19.4, 21.0, 23.7, 25.8, 13.9]]))
print(clf.score(X_test, y_test))


from sklearn.neural_network import MLPClassifier

clf1=MLPClassifier(activation='logistic',max_iter=1000)# 构造分类器实例
clf1.fit(X_train, y_train)
print(clf.predict([[120.0, 19.4, 21.0, 23.7, 25.8, 13.9]]))
print(clf.score(X_test, y_test))