from sklearn import datasets, neighbors, linear_model
import pandas as  pd
from sklearn.model_selection import train_test_split

advertise= pd.read_csv("advertising.csv")
X_train, X_test, y_train, y_test = train_test_split(advertise.loc[:,["Daily Time Spent on Site","Age","Area Income","Male"]], advertise.iloc[:,[-1]], test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=200)
logmodel.fit(X_train, y_train)

train_score = logmodel.score(X_train,y_train)
test_score = logmodel.score(X_test,y_test)

print(train_score,test_score)

y_pred_proba = logmodel.predict_proba(X_test)
print('sample of predict probability: {0}'.format(y_pred_proba[0]))#是0？还是1（概率）0在前1在后
print(y_pred_proba)
