import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
a = [[71],[68],[66],[67],[70],[71],[70],[73],[72],[65],[66]]
b = [69,64,65,63,65,62,65,64,66,59,62]
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

plt.scatter(a, b, label = 'real data')
plt.xlabel('bother height')
plt.ylabel('sister height')
plt.title('this is a demo')

reg = LinearRegression().fit(a, b)
y_pred = reg.predict(a)
plt.plot(a, y_pred, color='red', label = 'prediect')
plt.legend()                # 将标注显示出来
plt.show()
