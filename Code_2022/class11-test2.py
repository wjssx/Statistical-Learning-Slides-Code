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

status = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n_status = len(status)
m_obs = len(obs)
start_probability = np.array([0.2, 0.5, 0.3])
transition_probability = np.array([
    [0.5, 0.4, 0.1],
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

se = np.array([[0, 1, 0, 0, 1]]).T
logprob, box_index = model.decode(se, algorithm='viterbi')
print("颜色:", end="")
print(" ".join(map(lambda t: obs[t], [0, 1, 0, 0, 1])))
print("盒子:", end="")
print(" ".join(map(lambda t: status[t], box_index)))
print("概率值:", end="")
print(np.exp(logprob))  # 这个是因为在hmmlearn底层将概率进行了对数化，防止出现乘积为0的情况

status = ['盒子1', '盒子2', '盒子3']
obs = ['白球', '黑球']
n_status = len(status)
m_obs = len(obs)
start_probability = np.array([0.2, 0.5, 0.3])
transition_probability = np.array([
    [0.5, 0.4, 0.1],
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
seen = np.array([0, 1, 0])

# 观测序列的概率计算问题
# score函数返回的是以自然对数为底的对数概率值
# ln0.13022≈−2.0385
print(model.score(seen.reshape(-1, 1)))

print(np.exp(-1.81))
