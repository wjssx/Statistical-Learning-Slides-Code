import pandas as  pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
fish= pd.read_csv("fish-01.csv")
X_train, X_test, y_train, y_test = train_test_split(fish.iloc[:,1:], fish.iloc[:,0], test_size=0.4, random_state=0)



clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(clf.predict([[120.0, 19.4, 21.0, 23.7, 25.8, 13.9]]))

print(clf.score(X_test, y_test))


