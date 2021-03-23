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

    # 返回数据集和每个维度的名称
    return datasets, labels
from math import log

# 以 Outlook 为分界的熵

En_Sunny = -(2/5)*log(2/5,2) - (3/5)*log(3/5,2)
En_Overcast = -(4/4)*log(4/4,2)
En_Rainy = -(3/5)*log(3/5,2) - (2/5)*log(2/5,2)

# Outlook 联合熵
En_Outlook = 5/14*En_Sunny + 4/14*En_Overcast + 5/14*En_Rainy

print(En_Sunny,En_Overcast,En_Rainy)
print('联合熵:',En_Outlook)
# Outlook 的分裂信息度量 熵

IG=-(5/14)*log(5/14,2) - (9/14)*log(9/14,2)-En_Outlook
print("信息增益",IG)
OutLook = -5/14*log(5/14,2)-4/14*log(4/14,2)-5/14*log(5/14,2)
# Outlook 增益率
OutLook_Gain_Ratio = IG/OutLook

print(OutLook,OutLook_Gain_Ratio)

import numpy as  np


# 定义节点类 二叉树
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)


import pandas as pd

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    # 返回数据集和每个维度的名称
    return datasets, labels


datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)

print(dt.predict(['老年', '否', '否', '一般']))

