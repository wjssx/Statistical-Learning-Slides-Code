from math import exp

import numpy as np
from hmmlearn import hmm

status = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n_status = len(status)
m_obs = len(obs)
start_probability = np.array([0.2, 0.5, 0.3])
transition_probability = np.array([
    [0.5, 0.4, 0.1],      #盒子1到1，1到2，1到3的概率
    [0.2, 0.2, 0.6],
    [0.2, 0.5, 0.3]
])
emission_probalitity = np.array([
    [0.4, 0.6],
    [0.8, 0.2],
    [0.5, 0.5]
])

model = hmm.MultinomialHMM(n_components=n_status)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probalitity

# 预测问题
seen=np.array([0,1,0]) #白球，黑球，白球

# 观测序列的概率计算问题
# score函数返回的是以自然对数为底的对数概率值
# ln0.13022≈−2.0385
print(exp(model.score(seen.reshape(-1,1))))