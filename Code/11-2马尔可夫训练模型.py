import numpy as np
import hmmlearn.hmm as hmm

states = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n_states = len(states)
m_obs = len(obs)

model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.001)
X2 = np.array([
    [0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0]
])
model2.fit(X2)
print("输出根据数据训练出来的π")
print(model2.startprob_)
print("输出根据数据训练出来的A")
print(model2.transmat_)
print("输出根据数据训练出来的B")
print(model2.emissionprob_)
#从观测的结果反过去求盒子和球