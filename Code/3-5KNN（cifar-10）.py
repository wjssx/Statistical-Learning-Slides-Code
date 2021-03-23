from sklearn.neighbors import KNeighborsClassifier
import pickle
import cv2


def load(filename):

    with open(filename, 'rb') as fo:

        data = pickle.load(fo, encoding='latin1')

    return data
#读取第一个训练集——data_batch_1：
train = 'cifar-10-batches-py\data_batch_'
test=r'cifar-10-batches-py\test_batch'       #字符串前加r防止转义字符/t
print(test)
clf = KNeighborsClassifier("nn")

for i in  range(1,6): #从文件cifar-10-batches-py中读取data集1-5
    d=load(train+str(i))
    X, y = d["data"], d["labels"]
    X_train, y_train = X, y
    clf.fit(X_train, y_train)
    print("数据集" + str(i) + "训练完毕")
d=load(test)#从文件cifar-10-batches-py中读取test集
X, y = d["data"], d["labels"]
X_test, y_test = X, y

print(clf.score(X_test, y_test))