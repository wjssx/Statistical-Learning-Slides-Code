import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# bagging 算法
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)

in_score = bagging.score(X_train, y_train)
out_score = bagging.score(X_test, y_test)
print(in_score, out_score)

# RandomForest 算法
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=300, max_depth=2, random_state=0)
forest.fit(X_train, y_train)

in_score = forest.score(X_train, y_train)
out_score = forest.score(X_test, y_test)
print(in_score, out_score)

# Adaboost 算法

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
clf.fit(X_train, y_train)
in_score = clf.score(X_train, y_train)
out_score = clf.score(X_test, y_test)
print(in_score, out_score)

# 投票分类器
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

import numpy as np

clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
eclf1 = VotingClassifier(estimators=[
    ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
# print(eclf1.predict(X))
in_score = eclf1.score(X_train, y_train)
out_score = eclf1.score(X_test, y_test)
print(in_score, out_score)


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier


X, y = load_iris(return_X_y=True)
estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
     ('svr', make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42)))]
clf = StackingClassifier(
     estimators=estimators, final_estimator=LogisticRegression()
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, stratify=y, random_state=42
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))