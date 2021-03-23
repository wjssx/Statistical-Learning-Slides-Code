import pandas as pd
import numpy as np


Iteration=[]
w_0=[-1.5]
w_1=[0]
w_2=[2]
Training_Example=[]
x1=[]
x2=[]
Class=[]
s=[]
Action=[]

count=0



data = np.array([[0,1,-1],[2, 0,-1],[1,1,1]])
X, y = data[:,:-1], data[:,-1]

class Perceptron_Model:
    def __init__(self):
        self.w = np.array([0,2])
        self.b = -1.5
        self.l_rate = 1.0

        # self.data = data


    def sign(self, x, w, b):

        y = np.dot(x, w) + b
        s.append(y)
        return y

    # 随机梯度下降法
    def fit(self, X_train, y_train):
        count=1
        is_wrong = False
        while not is_wrong:


            wrong_count = 0
            for d in range(len(X_train)):
                if d==0:
                    Training_Example.append("a")
                elif d==1:
                    Training_Example.append("b")
                else:
                    Training_Example.append("c")
                X = X_train[d]
                x1.append(X[0])
                x2.append(X[1])
                y = y_train[d]
                if y<0:
                    Class.append("-")
                else:
                    Class.append("+")
                if y * self.sign(X, self.w, self.b) <= 0:
                    Iteration.append(count)
                    count = count + 1
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    w_1.append(self.w[0])
                    w_2.append(self.w[1])
                    self.b = self.b + self.l_rate * y
                    w_0.append(self.b)
                    if(y>0):Action.append("Add")
                    else:Action.append("Subtract")
                    wrong_count += 1
                else:
                    Iteration.append(count)
                    count = count + 1
                    w_1.append(self.w[0])
                    w_2.append(self.w[1])
                    w_0.append(self.b)
                    Action.append("None")

            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'

    def score(self):
        pass
print()
perceptron = Perceptron_Model()
perceptron.fit(X, y)
print(count)

record = {
    'Iteration':Iteration,
    'w_0':w_0[0:12],
    'w_1':w_1[0:12],
    'w_2':w_2[0:12],
'Training_Example':Training_Example,
'x1':x1,
'x2':x2,
'Class':Class,
's=w_0+w_1x_1+w_2x_2':s,
'Action':Action
}

print(record)
frame = pd.DataFrame(record)
frame.to_csv(path_or_buf="tmp.csv",index=False)
