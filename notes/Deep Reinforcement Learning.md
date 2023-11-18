# Deep Reinforcement Learning

## 1.terminology

**Random Variable** ：随机变量

X->随机变量，x->观测值

**Probability Density Function**（PDF）：概率密度函数

discrete 离散    求和       $\sum_{x\in x}p(x) = 1$

continuous 连续    定积分  $\int_{x}{p(x)}{\rm d}x = 1$

**Expectation** ：期望

**Random Sampling**：随机抽样



**state** : s

**action** : a

**agent**

**policy** ：策略

**策略函数**   policy function $\pi$ : (s, a)        $\pi(a|s) = p(A=a|S=s)$ 给定状态s做出a的概率密度

action最好为随机抽样得到，否则为具体的策略，别人可以针对此制定必胜策略

**reward** : R  奖励                强化学习目标：尽量大的reward

**state transition** : p 状态转移      $p(s'|s,a)=P(S'=s'|S=s,A=a)$    当前状态s和采取行动a，下一状态为s'的概率

状态转移随机性   由环境决定    old state ---(action)--->new state



**Return** : U    未来的累计奖励 cumulative future reward  

$U_t=R_t+R_{t+1}+R_{t+2}+R_{t+3}+...$ 

**Discounted return** : 折扣汇报（未来奖励的折扣  未来奖励<=现在的奖励）

折扣$\gamma \in [0,1]$ 在0-1之间   $\gamma =1$ 为没有折扣， $\gamma$ 为超参数

$U_t=R_t+\gamma R_{t+1}+\gamma ^2R_{t+2}+\gamma ^3R_{t+3}+...$ 

**Action-value function** : 动作价值函数 $Q_\pi (s_t,a_t)=E[U_t|S_t=s_t,A_t=a_t]$

**Optimal action-value function** : 最优动作价值函数 $Q^*=max_\pi Q_\pi(s_t,a_t)$

与$\pi$ 无关

**State-value function** : 状态价值函数 $Q_\pi$ 的期望，与a无关

$V_\pi (s_t)=E_A[Q_\pi(s_t,A)]=\sum_a\pi(a|s_t)Q_\pi(s_t,a)$

判断状态好坏，快输了或快赢了



选择行动

策略学习     随机抽样 $a_t$ ~ $\pi(.|s_t)$

价值学习     $a_t=argmax_aQ^*(s_t,a)$



## Value-based reinforcement learning 价值学习

DQN   Deep Q Network

用神经网络Q(s,a;w)  w是神经网络参数 来approximate近似 Q*(s,a)

训练DQN-> Temporal Difference (TD) Learning

流程：

1. 观测当前state $S_t=s_T$ , 已经执行的action $A_t=a_t$ 

2. 预测value，DQN输出对action $a_t$ 的打分  $q_t=Q(s_t,a_t;w_t)$

3. 反向传播求梯度  $d_t=\frac{\partial Q(s_t,a_t;w)}{\partial w}|w=w_t$

4. 由于已经执行了$a_t$ ，更新状态$s_{t+1}$ 并得到reward $r_t$

5. compute计算 TD target   $y_t=r_t+\gamma max_a Q(s_{t+1},a;w_t)$

   Loss: $L(w) = \frac{1}{2}[q_t-y_t]^2$

6. 梯度下降更新模型参数  $w_{t+1}=w_t-\alpha (q_t-y_t)d_t$ , $\alpha$ 是学习率

## Policy-based reinforcement learning 策略学习

策略函数input是当前状态s，输出为全部action的概率分布，通过随机抽样得到action

policy network 策略神经网络 $\pi(a|s;\theta)$ 来approximate $\pi(a|s)$， $\theta$ 为网络参数

约束：$\sum_{a\in A}\pi(a|s;\theta)=1$       采用softmax激活函数

Softmax是一种激活函数，它可以将一个数值向量归一化为一个概率分布向量，且各个概率之和为1。Softmax可以用来作为神经网络的最后一层，用于多分类问题的输出。Softmax层常常和交叉熵损失函数一起结合使用。

二分类问题：采用sigmod函数（又称logistic函数），将$(-\infty, \infty)$ 范围内的数值映射为一个$(0,1)$ 区间的数值，用来表示概率
$$
g(z)=\frac{1}{1+e^{-z}}
$$
多分类问题：采用softmax函数
$$
softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$
近似状态价值函数

$V_\pi (s_t)=E_A[Q_\pi(s_t,A)]=\sum_a\pi(a|s_t)Q_\pi(s_t,a)$  --->

$V_\pi (s_t;\theta)=\sum_a\pi(a|s_t;\theta)Q_\pi(s_t,a)$

策略网络越好，V的值越大，即改变$\theta$ 使V最大,即使$J(\theta)$ 最大， $J(\theta) = E_s[V(S;\theta)]$

Policy gradient 策略梯度



## Actor-Critic Methods

策略学习-->actor 与 价值学习-->critic 结合

$V_\pi (s)=\sum_a\pi(a|s)Q_\pi(s,a) \approx \sum_a \pi(a|s;\theta)q(s,a;w)$

算法流程：

每一轮迭代里只有一次action，观测一次reward，更新一次神经网络参数

1. 观测state $s_t$ 和 随机抽样得到action $a_t$ ~ $\pi(.|s_t;\theta_t)$
2. agent执行action $a_t$ ；从environment得到new state $s_{t+1}$ 和reward $r_t$
3. 随机抽样action $\hat a_{t+1}$ ~ $\pi(.|s_{t+1};\theta_t)$   ($\hat a_{t+1}$ 为假想的动作，实际不执行)
4. 计算价值网络打分 $q_t=q(s_t,a_t;w_t)$ 和 $q_{t+1}=q(s_{t+1}, \hat a_{t+1};w_t)$
5. 计算TD error ： $\delta_t=q_t-(r_t+\gamma q_{t+1})$ 其中 TD target 为 $r_t+\gamma q_{t+1}$
6. 对价值网络求导 $d_{w,t}=\frac{\partial q(s_t,a_t;w)}{\partial w}|w=w_t$
7. 梯度下降更新价值网络 $w_{t+1}=w_t-\alpha \delta_t d_{w,t}$      ，$\alpha$ 为学习率
8. 对策略网络求导 $d_{\theta,t}=\frac{\partial log\pi(a_t|s_t;\theta)}{\partial \theta}|theta=\theta_t$ 
9. 梯度上升更新策略网络 $\theta_{t+1}=\theta_t-\beta q_t d_{\theta,t}$   ，$\beta$ 为学习率

通常，使用baseline时，9中为$\theta_{t+1}=\theta_t-\beta \delta_t d_{\theta,t}$ ，均正确

好的baseline可以降低方差，让算法收敛更快，任何接近$q_t$ 的且不为$a_t$ 的函数的数均可作为baseline，例如用TD target为baseline

