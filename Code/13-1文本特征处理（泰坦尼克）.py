import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('onehot/train.csv')
train.head()


train.info()


data = train
data['Died']= 1 - data['Survived']
plt.show(data.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',stacked=True))


encoder = OneHotEncoder(sparse=False)
En_ec = encoder.fit_transform(train[['Sex']])
En_ec = pd.DataFrame(En_ec)
train_new = pd.concat([train,En_ec],axis=1)





