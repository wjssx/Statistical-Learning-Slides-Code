
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt



# data
def create_data(col = 2):
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:col], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



#svm = SVM(max_iter=400)


#svm.fit(X_train, y_train)


from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train)

# sklearn.svm.SVC
'''
*(C
  =1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None) *

参数：

- C：C - SVC的惩罚参数C?默认值是1
.0

C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

- kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

– 线性：u'v

– 多项式：(gamma * u'*v + coef0)^degree

       – RBF函数：exp(-gamma | u-v | ^ 2)

– sigmoid：tanh(gamma * u'*v + coef0)

               - decision_function_shape: ‘ovo’, ‘ovr’, default =’ovr’

Whether
to
return a
one - vs - rest(‘ovr’) decision
function
of
shape(n_samples, n_classes) as all
other
classifiers, or the
original
one - vs - one(‘ovo’) decision
function
of
libsvm
which
has
shape(n_samples, n_classes * (n_classes - 1) / 2).However, one - vs - one(‘ovo’) is always
used as multi -


class strategy.



a.一对多法（one - versus - rest, 简称1 - v - r
SVMs）。训练时依次把某个类别的样本归为一类, 其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。 

b.一对一法（one - versus - one, 简称1 - v - 1
SVMs）。其做法是在任意两类样本之间设计一个SVM，因此k个类别的样本就需要设计k(k - 1) / 2
个SVM。当对一个未知样本进行分类时，最后得票最多的类别即为该未知样本的类别。Libsvm中的多类分类就是根据这个方法实现的。

- degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。


- gamma ： ‘rbf’, ‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1 / n_features

- coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。


- probability ：是否采用概率估计？.默认为False

- shrinking ：是否采用shrinking
heuristic方法，默认为true

- tol ：停止训练的误差值大小，默认为1e - 3

- cache_size ：核函数cache缓存大小，默认为200

- class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight * C(C - SVC中的C)

- verbose ：允许冗余输出？


- max_iter ：最大迭代次数。-1
为无限制。


- decision_function_shape ：‘ovo’, ‘ovr’ or None, default = None3

- random_state ：数据洗牌时的种子值，int值

主要调节的参数有：C、kernel、degree、gamma、coef0。
'''