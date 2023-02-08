def create_data():
    datasets = [[1, 'Sunny', 'Hot', 'High', 'Weak', 'No'],
                [2, 'Sunny', 'Hot', 'High', 'Strong', 'No'],
                [3, 'Overcast', 'Hot', 'High', 'Weak', 'Yes'],
                [4, 'Rainy', 'Mild', 'High', 'Weak', 'Yes'],
                [5, 'Rainy', 'Cool', 'Normal', 'Weak', 'Yes'],
                [6, 'Rainy', 'Cool', 'Normal', 'Strong', 'No'],
                [7, 'Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
                [8, 'Sunny', 'Mild', 'High', 'Weak', 'No'],
                [9, 'Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
                [10, 'Rainy', 'Mild', 'Normal', 'Weak', 'Yes'],
                [11, 'Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
                [12, 'Overcast', 'Mild', 'High', 'Strong', 'Yes'],
                [13, 'Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
                [14, 'Rainy', 'Mild', 'High', 'Strong', 'No'],
                ]

    labels = ['Day', 'OutLook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
    return datasets, labels
    # 返回数据集和每个维度的名称


import pandas as pd
from math import log2

datasets, labels = create_data()

train_data = pd.DataFrame(datasets, columns=labels)

print(train_data)

# 以 Outlook 为分界的熵
En_Sunny = -(2 / 5) * log2(2 / 5) - (3 / 5) * log2(3 / 5)
En_Overcast = -(4 / 4) * log2(4 / 4)
En_Rainy = -(3 / 5) * log2(3 / 5) - (2 / 5) * log2(2 / 5)

# Outlook 熵
En_Outlook = 5 / 14 * En_Sunny + 4 / 14 * En_Overcast + 5 / 14 * En_Rainy

print(En_Outlook)

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print(clf.predict([[2., 2.]]))
print(clf.predict_proba([[2., 2.]]))