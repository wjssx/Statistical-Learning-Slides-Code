# 对决策树模型在数字样本上的测试结果做评估
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
digits = load_digits()
X, y = digits.data, digits.target


X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train,)
print(clf.score(X_test, Y_test))


y_pred=clf.predict(X_test)
from sklearn.metrics import precision_score
print(precision_score(Y_test,y_pred,average=None))


from sklearn.metrics import recall_score
print(recall_score(Y_test,y_pred,average=None))


from sklearn.preprocessing import label_binarize   #给数字数据集做归类化

# Use label_binarize to be multi-label like settings
Y_test_b = label_binarize(Y_test, classes=[0, 1, 2,3,4,5,6,7,8,9])
Y_pred_b= label_binarize(y_pred, classes=[0, 1, 2,3,4,5,6,7,8,9])
n_classes = Y_test_b.shape[1]


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):                  #计算每个标签的召回率
    precision[i], recall[i], _ = precision_recall_curve(Y_test_b[:, i],
                                                        Y_pred_b[:, i])
    average_precision[i] = average_precision_score(Y_test_b[:, i],Y_pred_b[:, i])

lines = []
labels = []

colors = ['red','blue','green','pink','gold','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal']

# precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test_b.ravel(),
#     Y_pred_b.ravel())
# average_precision["micro"] = average_precision_score(Y_test_b,Y_pred_b,  #平均召回率
#                                                      average="micro")
# print('Average precision score, micro-averaged over all classes: {0:0.2f}'
#       .format(average_precision["micro"]))

plt.figure(figsize=(7, 8))
for i, color in zip(range(n_classes), colors):          #在图里画出每个标签的召回率
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.0), prop=dict(size=9))

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
disp = plot_confusion_matrix(clf, X_test, Y_test)
disp.figure_.suptitle("Confusion Matrix")


print(classification_report(Y_test, y_pred))

plt.show()