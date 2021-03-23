
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#% matplotlib
#inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter

dia = pd.read_csv("diabetes.csv")
df = pd.DataFrame(dia)

print(df)


data = np.array(df.iloc[:767, [0,1,2,3,4,6,-1]])
print(data)
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



from sklearn.neighbors import KNeighborsClassifier
#
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
#
#
print(clf_sk.score(X_test, y_test))
#
#
