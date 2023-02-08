# Example 1
import numpy as np

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

from sklearn.svm import SVC

# clf = SVC(gamma='auto')

clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
          max_iter=-1, probability=False, random_state=None, shrinking=True,
          tol=0.001, verbose=False)  # 可以根据前面介绍的参数，做出相应改变观察结果变化

clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))

print(clf.support_vectors_)
print(clf.dual_coef_, clf.coef_, clf.intercept_)
