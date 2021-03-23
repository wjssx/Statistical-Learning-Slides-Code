
from sklearn import svm
import pickle

from sklearn.model_selection import GridSearchCV


def load(filename):

    with open(filename, 'rb') as fo:

        data = pickle.load(fo, encoding='latin1')

    return data
#读取第一个训练集——data_batch_1：
train = 'cifar-10-batches-py\data_batch_'
test=r'cifar-10-batches-py\test_batch'       #字符串前加r防止转义字符/t
print(test)



#parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), 'C':[0.1,1,10,100]}
#classifier=svm.SVC()
#classifier.n_jobs=-1
#print("1")
#clf = GridSearchCV(classifier, parameters)
clf=svm.SVC()
for j in  range(1,6): #从文件cifar-10-batches-py中读取data集1-5
        d=load(train+str(j))
        print("数据集"+str(j)+"训练完毕")
        X, y = d["data"], d["labels"]
        X_train, y_train = X, y
        clf.fit(X_train, y_train,)

d=load(test)#从文件cifar-10-batches-py中读取test集
X, y = d["data"], d["labels"]
X_test, y_test = X, y
clf.fit(X_test, y_test,)


print(clf.score)
