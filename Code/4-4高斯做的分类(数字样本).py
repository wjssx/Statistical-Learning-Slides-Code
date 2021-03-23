
from sklearn.datasets import load_digits
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
digits = load_digits()
X, y = digits.data, digits.target


print(digits.data.shape)

import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(digits.images[0])
# plt.show()

fig=plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

for i in range(64):
    ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    #用目标值标记图像
    ax.text(0,7,str(digits.target[i]))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB(priors=None, var_smoothing=1e-09)
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))