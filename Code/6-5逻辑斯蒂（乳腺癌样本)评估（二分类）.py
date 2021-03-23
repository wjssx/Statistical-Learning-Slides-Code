
# 载入数据
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape: {0}; no. positive: {1}; no. negative: {2}'.format(
    X.shape, y[y==1].shape[0], y[y==0].shape[0]))
print(cancer.data[0])


cancer.feature_names


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=3000)
logmodel.fit(X_train, y_train)

train_score = logmodel.score(X_train,y_train)
test_score = logmodel.score(X_test,y_test)

#print(train_score,test_score)

X_output=logmodel.predict(X_test)
from sklearn.metrics import precision_score
print(precision_score(y_test,X_output,average=None))

# （查全率）召回率  The best value is 1 and the worst value is 0.
# 就是所有准确的条目有多少被检索出来了。
from sklearn.metrics import recall_score
print(recall_score(y_test,X_output,average=None))




from sklearn.metrics import plot_precision_recall_curve

pr = plot_precision_recall_curve(logmodel,X_test,y_test)

#print(pr)

from sklearn.metrics import plot_roc_curve

roc = plot_roc_curve(logmodel, X_test, y_test)


# from sklearn.metrics import roc_curve,auc
# fpr, tpr, thresholds = roc_curve(y_test, X_output)
# print(auc(fpr, tpr))


from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
disp = plot_confusion_matrix(logmodel, X_test, y_test)  #混淆矩阵
disp.figure_.suptitle("Confusion Matrix")
#print("Confusion matrix:\n%s" % disp.confusion_matrix)

print(classification_report(y_test, X_output))

plt.show()